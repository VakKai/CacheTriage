#pragma once

#include "cachetriage/cache.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>

namespace cachetriage {

struct ReuseFeatures {
    std::uint64_t last_access;
    std::uint64_t resident_hits;

    void record_hit(std::uint64_t request) noexcept {
        last_access = request;
        ++resident_hits;
    }
};

struct ReuseModel {
    double bias;
    double age_weight;
    double hits_weight;

    double score(std::uint64_t request, const ReuseFeatures& features) const noexcept {
        return bias
            + age_weight * log_count(request - features.last_access)
            + hits_weight * log_count(features.resident_hits);
    }

    static constexpr std::size_t shared_table_bytes() { return sizeof(log_counts_); }

    static double log_count(std::uint64_t count) noexcept {
        return count < log_counts_.size() ? log_counts_[count] : std::log1p(static_cast<double>(count));
    }

private:
    // Common counts use the same log1p values without recomputing them.
    inline static const std::array<double, 256> log_counts_ = [] {
        std::array<double, 256> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = std::log1p(static_cast<double>(i));
        }
        return values;
    }();
};

enum class GateRule { always, age_gap, victim_hits };

struct ReuseGate {
    GateRule rule = GateRule::always;
    std::uint64_t threshold = 0;

    const char* threshold_field() const {
        return rule == GateRule::victim_hits ? "gate_min_victim_hits" : "gate_max_age_gap_reads";
    }
};

// Shares the shortlist decision between replay and the backing-store demo.
class ReuseSelector {
public:
    ReuseSelector(ReuseModel model, ReuseGate gate)
        : model_(model), gate_(gate) {}

    template <typename Lookup>
    std::size_t select(const EvictionShortlist& shortlist, std::uint64_t request, Lookup lookup) noexcept {
        if (!should_score(shortlist, lookup)) {
            return 0;
        }
        ++calls_;
        std::size_t victim = 0;
        double lowest_score = model_.score(request, lookup(0));
        for (std::size_t i = 1; i < shortlist.size; ++i) {
            const double score = model_.score(request, lookup(i));
            if (score < lowest_score) {
                lowest_score = score;
                victim = i;
            }
        }
        swaps_ += victim != 0;
        return victim;
    }

    std::uint64_t swaps() const { return swaps_; }
    std::uint64_t calls() const { return calls_; }

private:
    template <typename Lookup>
    bool should_score(const EvictionShortlist& shortlist, Lookup lookup) const noexcept {
        if (gate_.rule == GateRule::always) {
            return true;
        }
        if (gate_.threshold == 0 || shortlist.size < 2) {
            return false;
        }
        const auto& victim = lookup(0);
        const auto& neighbor = lookup(1);
        if (gate_.rule == GateRule::victim_hits) {
            return victim.resident_hits >= gate_.threshold && neighbor.resident_hits == 0;
        }
        // Both ages would subtract from the same request number.
        const auto age_gap = std::max(victim.last_access, neighbor.last_access)
                           - std::min(victim.last_access, neighbor.last_access);
        return victim.resident_hits == neighbor.resident_hits && age_gap <= gate_.threshold;
    }

    ReuseModel model_;
    ReuseGate gate_;
    std::uint64_t swaps_ = 0;
    std::uint64_t calls_ = 0;
};

inline std::optional<ReuseModel> load_reuse_model(const char* path) {
    std::ifstream input(path);
    std::string version, bias_name, age_name, hits_name;
    ReuseModel model{};
    if (!(input >> version >> bias_name >> model.bias >> age_name >> model.age_weight
                >> hits_name >> model.hits_weight) ||
        version != "cachetriage_logistic_v1" || bias_name != "bias" ||
        age_name != "log1p_age_reads" || hits_name != "log1p_resident_hits" ||
        !std::isfinite(model.bias) || !std::isfinite(model.age_weight) ||
        !std::isfinite(model.hits_weight)) {
        return std::nullopt;
    }
    input >> std::ws;
    if (!input.eof() || input.bad()) {
        return std::nullopt;
    }
    return model;
}

}  // namespace cachetriage
