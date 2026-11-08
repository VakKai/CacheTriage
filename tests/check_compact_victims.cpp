#include "pairwise_model.hpp"
#include "trace_reader.hpp"

#include <exception>
#include <iostream>

using namespace cachetriage;

namespace {

void verify(const char* model_path, const char* trace_path) {
    const auto model = load_pairwise_models(model_path).logistic;
    Cache<EvictionHistory> original(128, Policy::sieve);
    Cache<ReuseFeatures> compact(128, Policy::sieve);
    PairwiseSelector selector(model);
    std::uint64_t request = 0, evictions = 0;
    for_each_request(trace_path, [&](Key key) {
        ++request;
        auto* old_entry = original.find(key);
        auto* new_entry = compact.find(key);
        if ((old_entry == nullptr) != (new_entry == nullptr)) {
            throw std::runtime_error("hit differs");
        }
        if (old_entry) {
            old_entry->record_hit(request);
            new_entry->record_hit(request);
            return;
        }
        const auto old_select = [&](const EvictionShortlist& shortlist, const auto& values) noexcept {
            auto best_score = model.cutoff;
            std::size_t best = 0;
            for (std::size_t i = 1; i < shortlist.size; ++i) {
                // The original full-history logistic score uses slots 0, 1, 4, and 5.
                const auto f = pair_features(*values[0], *values[i], request);
                const auto& w = model.weights;
                const auto score = model.intercept + w[0] * f[0] + w[1] * f[1] + w[2] * f[4] + w[3] * f[5];
                if (score > best_score) {
                    best_score = score;
                    best = i;
                }
            }
            return best;
        };
        const auto new_select = [&](const EvictionShortlist& shortlist, const auto& values) noexcept {
            return selector.select(shortlist, request, [&](std::size_t i) -> const ReuseFeatures& {
                return *values[i];
            });
        };
        const auto old_victim = original.insert(key, {request, request}, nullptr, old_select);
        const auto new_victim = compact.insert(key, {request, 0}, nullptr, new_select);
        if (old_victim != new_victim) {
            throw std::runtime_error("victim differs at request " + std::to_string(request));
        }
        evictions += old_victim.has_value();
    });
    std::cout << "{\"requests\": " << request << ", \"matching_evictions\": " << evictions
              << ", \"original_history_bytes\": " << sizeof(EvictionHistory)
              << ", \"compact_history_bytes\": " << sizeof(ReuseFeatures) << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: check_compact_victims <pairwise-model> <trace>\n";
        return 1;
    }
    try {
        verify(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
