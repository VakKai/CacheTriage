#pragma once

#include "eviction_history.hpp"
#include "reuse_model.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cachetriage {

using PairFeatures = std::array<double, 8>;

inline PairFeatures pair_features(const EvictionHistory& victim, const EvictionHistory& alternative,
                                  std::size_t request) noexcept {
    PairFeatures result{};
    const auto counts = [request](const EvictionHistory& entry) {
        return std::array<std::size_t, 4>{request - entry.last, entry.hits, request - entry.admitted,
                                         entry.previous ? entry.last - entry.previous : 0};
    };
    const auto base = counts(victim);
    const auto other = counts(alternative);
    for (std::size_t i = 0; i < base.size(); ++i) {
        result[i] = ReuseModel::log_count(base[i]);
        result[4 + i] = ReuseModel::log_count(other[i]) - result[i];
    }
    return result;
}

struct LinearOverride {
    double cutoff;
    double intercept;
    std::array<double, 4> weights;

    static constexpr std::size_t bytes() {
        return sizeof(LinearOverride) + ReuseModel::shared_table_bytes();
    }

    template <typename Lookup>
    std::size_t select(std::size_t count, std::uint64_t request, Lookup lookup) const noexcept {
        const auto& victim = lookup(0);
        const double age = ReuseModel::log_count(request - victim.last_access);
        const double hits = ReuseModel::log_count(victim.resident_hits);
        const double base = intercept + weights[0] * age + weights[1] * hits;
        double best_score = cutoff;
        std::size_t best = 0;
        for (std::size_t i = 1; i < count; ++i) {
            const auto& alternative = lookup(i);
            const double age_delta = ReuseModel::log_count(request - alternative.last_access) - age;
            const double hit_delta = ReuseModel::log_count(alternative.resident_hits) - hits;
            const double score = base + weights[2] * age_delta + weights[3] * hit_delta;
            if (score > best_score) {
                best = i;
                best_score = score;
            }
        }
        return best;
    }
};

class PairwiseSelector {
public:
    explicit PairwiseSelector(const LinearOverride& model) : model_(model) {}

    template <typename Lookup>
    std::size_t select(const EvictionShortlist& shortlist, std::uint64_t request, Lookup lookup) noexcept {
        if (shortlist.size < 2) {
            return 0;
        }
        ++calls_;
        return model_.select(shortlist.size, request, lookup);
    }

    std::uint64_t calls() const { return calls_; }

private:
    const LinearOverride& model_;
    std::uint64_t calls_ = 0;
};

struct ShallowTree {
    struct Node {
        double threshold;
        double value;
        std::int8_t feature;
        std::int8_t left;
        std::int8_t right;
    };
    std::array<Node, 7> nodes{};

    double score(const std::array<float, 8>& features) const noexcept {
        std::size_t i = 0;
        while (nodes[i].left >= 0) {
            const auto& node = nodes[i];
            i = features[node.feature] <= node.threshold ? node.left : node.right;
        }
        return nodes[i].value;
    }
};

struct BoostedOverride {
    double cutoff;
    double initial_logit;
    double learning_rate;
    std::vector<ShallowTree> trees;

    double score(const PairFeatures& features) const noexcept {
        std::array<float, 8> inputs{};
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = static_cast<float>(features[i]);
        }
        double result = initial_logit;
        for (const auto& tree : trees) {
            result += learning_rate * tree.score(inputs);
        }
        return result;
    }

    std::size_t bytes() const { return sizeof(*this) + trees.capacity() * sizeof(ShallowTree); }
};

struct PairwiseModels {
    LinearOverride logistic;
    BoostedOverride boosted;
};

inline double read_finite(std::istream& input) {
    double value = 0;
    if (!(input >> value) || !std::isfinite(value)) {
        throw std::runtime_error("expected a finite pairwise model coefficient");
    }
    return value;
}

inline double read_cutoff(std::istream& input) {
    const double probability = read_finite(input);
    if (probability < 0 || probability > 1) {
        throw std::runtime_error("invalid pairwise probability threshold");
    }
    if (probability == 0 || probability == 1) {
        return probability == 0 ? -std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::infinity();
    }
    return std::log(probability / (1 - probability));
}

inline PairwiseModels load_pairwise_models(const char* path) {
    std::ifstream input(path);
    std::string version;
    if (!(input >> version) || version != "cachetriage_pairwise_replay_v1") {
        throw std::runtime_error("cannot read pairwise model: " + std::string(path));
    }
    PairwiseModels result{};
    result.logistic.cutoff = read_cutoff(input);
    result.logistic.intercept = read_finite(input);
    for (double& weight : result.logistic.weights) {
        weight = read_finite(input);
    }
    result.boosted.cutoff = read_cutoff(input);
    result.boosted.initial_logit = read_finite(input);
    result.boosted.learning_rate = read_finite(input);
    std::size_t tree_count = 0;
    if (!(input >> tree_count) || tree_count == 0 || tree_count > 1000 ||
        result.boosted.learning_rate <= 0) {
        throw std::runtime_error("invalid boosted model settings");
    }
    result.boosted.trees.resize(tree_count);
    for (auto& tree : result.boosted.trees) {
        int count = 0;
        if (!(input >> count) || count < 1 || count > static_cast<int>(tree.nodes.size())) {
            throw std::runtime_error("invalid tree node count");
        }
        for (int i = 0; i < count; ++i) {
            int feature = 0, left = 0, right = 0;
            if (!(input >> feature >> left >> right)) {
                throw std::runtime_error("missing tree node");
            }
            const bool leaf = left == -1 && right == -1;
            if (!leaf && (feature < 0 || feature >= 8 || left <= i || right <= i ||
                          left >= count || right >= count || left == right)) {
                throw std::runtime_error("invalid tree node");
            }
            tree.nodes[i] = {read_finite(input), read_finite(input),
                             static_cast<std::int8_t>(leaf ? -1 : feature),
                             static_cast<std::int8_t>(left), static_cast<std::int8_t>(right)};
        }
    }
    input >> std::ws;
    if (!input.eof() || input.bad()) {
        throw std::runtime_error("unexpected trailing pairwise model data");
    }
    return result;
}

}  // namespace cachetriage
