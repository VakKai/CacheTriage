#include "cache_budget.hpp"
#include "pairwise_model.hpp"
#include "replay_support.hpp"
#include "trace_reader.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using namespace cachetriage;
using namespace cachetriage::replay;

enum class Kind { sieve, eight_hit, history_rule, logistic, boosted };

constexpr std::array<const char*, 5> policy_names{
    "sieve", "eight_hit_gate", "history_rule", "pairwise_logistic", "pairwise_boosted",
};

const char* name(Kind kind) { return policy_names[static_cast<std::size_t>(kind)]; }

template <template <typename> class Allocator = std::allocator>
class PairwiseReplay {
public:
    PairwiseReplay(Kind kind, std::size_t capacity, const PairwiseModels& models,
                   Allocator<EvictionHistory> allocator)
        : stats_{name(kind)}, cache_(capacity, Policy::sieve, allocator), models_(models), kind_(kind) {}

    void read(Key key) {
        const auto request = stats_.requests() + 1;
        if (auto* entry = cache_.find(key)) {
            ++stats_.hits;
            entry->record_hit(request);
            return;
        }
        ++stats_.misses;
        const auto select = [&](const EvictionShortlist& shortlist, const auto& values) noexcept {
            const auto rank = select_victim(shortlist, values, request);
            swaps_ += rank != 0;
            return rank;
        };
        stats_.evictions += cache_.insert(key, {request, request}, nullptr, select).has_value();
    }

    void print_json(std::ostream& out) const { stats_.print_json(out, swaps_, calls_); }

private:
    using Values = typename Cache<EvictionHistory, Allocator>::ShortlistValues;

    std::size_t select_victim(const EvictionShortlist& shortlist, const Values& values,
                              std::size_t request) noexcept {
        if (shortlist.size < 2) {
            return 0;
        }
        const auto& victim = *values[0];
        std::size_t best = 0;
        if (kind_ == Kind::history_rule) {
            // Preserve a key within its last reuse interval if an older, less-hit
            // alternative exists. Prefer the least recently accessed eligible key.
            if (victim.previous && request - victim.last <= victim.last - victim.previous) {
                for (std::size_t i = 1; i < shortlist.size; ++i) {
                    if (values[i]->hits < victim.hits && values[i]->last < values[best]->last) {
                        best = i;
                    }
                }
            }
            return best;
        }

        ++calls_;
        double best_score = models_.boosted.cutoff;
        for (std::size_t i = 1; i < shortlist.size; ++i) {
            const auto score = models_.boosted.score(pair_features(victim, *values[i], request));
            if (score > best_score) {
                best = i;
                best_score = score;
            }
        }
        return best;
    }

    ReplayStats stats_;
    Cache<EvictionHistory, Allocator> cache_;
    const PairwiseModels& models_;
    Kind kind_;
    std::uint64_t calls_ = 0;
    std::uint64_t swaps_ = 0;
};

template <template <typename> class Allocator = std::allocator>
class CompactLogisticReplay {
public:
    CompactLogisticReplay(std::size_t capacity, const LinearOverride& model, Allocator<ReuseFeatures> allocator)
        : cache_(capacity, Policy::sieve, allocator), selector_(model) {}

    void read(Key key) {
        const auto request = stats_.requests() + 1;
        if (auto* entry = cache_.find(key)) {
            ++stats_.hits;
            entry->record_hit(request);
            return;
        }
        ++stats_.misses;
        const auto select = [&](const EvictionShortlist& shortlist, const auto& values) noexcept {
            const auto rank = selector_.select(shortlist, request,
                                               [&](std::size_t i) -> const ReuseFeatures& { return *values[i]; });
            swaps_ += rank != 0;
            return rank;
        };
        stats_.evictions += cache_.insert(key, {request, 0}, nullptr, select).has_value();
    }

    std::size_t misses() const { return stats_.misses; }
    void print_json(std::ostream& out) const { stats_.print_json(out, swaps_, selector_.calls()); }

private:
    ReplayStats stats_{name(Kind::logistic)};
    Cache<ReuseFeatures, Allocator> cache_;
    PairwiseSelector selector_;
    std::uint64_t swaps_ = 0;
};

template <typename Value, template <typename> class Allocator>
Allocator<Value> allocator_for(AllocationTally* tally) {
    if constexpr (std::is_same_v<Allocator<Value>, CountingAllocator<Value>>) {
        return Allocator<Value>(*tally);
    } else {
        return {};
    }
}

template <template <typename> class Allocator = std::allocator, typename Use>
void with_player(Kind kind, std::size_t capacity, ReuseModel reuse, const PairwiseModels& models,
                 Use use, AllocationTally* tally = nullptr) {
    if (kind == Kind::sieve) {
        Replay<Allocator> player(name(kind), capacity, Policy::sieve,
                                 allocator_for<std::monostate, Allocator>(tally));
        use(player, sizeof(Replay<>), 0);
    } else if (kind == Kind::eight_hit) {
        LearnedReplay<Allocator> player(capacity, reuse, {GateRule::victim_hits, 8},
                                         allocator_for<ReuseFeatures, Allocator>(tally), name(kind));
        use(player, sizeof(LearnedReplay<>), ReuseModel::shared_table_bytes());
    } else if (kind == Kind::logistic) {
        CompactLogisticReplay<Allocator> player(capacity, models.logistic,
                                                allocator_for<ReuseFeatures, Allocator>(tally));
        use(player, sizeof(CompactLogisticReplay<>), LinearOverride::bytes());
    } else {
        PairwiseReplay<Allocator> player(kind, capacity, models, allocator_for<EvictionHistory, Allocator>(tally));
        const auto model_bytes = kind == Kind::boosted
            ? models.boosted.bytes() + ReuseModel::shared_table_bytes() : 0;
        use(player, sizeof(PairwiseReplay<>), model_bytes);
    }
}

CacheMemory prefill_memory(Kind kind, std::size_t capacity, ReuseModel reuse, const PairwiseModels& models) {
    AllocationTally tally;
    CacheMemory memory;
    with_player<CountingAllocator>(kind, capacity, reuse, models,
        [&](auto& player, std::size_t object_bytes, std::size_t model_bytes) {
            for (std::size_t key = 0; key < capacity; ++key) {
                player.read(key);
            }
            memory = {object_bytes, tally.peak, model_bytes};
        }, &tally);
    return memory;
}

void scan(const std::vector<Key>& requests, std::size_t max_capacity, std::size_t target,
          ReuseModel reuse, const PairwiseModels& models) {
    // Miss counts can rise at a larger capacity, so test each capacity in order.
    for (std::size_t capacity = 0; capacity <= max_capacity; ++capacity) {
        CompactLogisticReplay<> player(capacity, models.logistic, {});
        replay_keys(player, requests);
        if (player.misses() <= target) {
            std::cout << "{\"capacity_items\": " << capacity << ", \"misses\": " << player.misses()
                      << ", \"peak_prefill_bytes\": "
                      << prefill_memory(Kind::logistic, capacity, reuse, models).total() << "}\n";
            return;
        }
        if (capacity == max_capacity) {
            break;
        }
    }
    std::cout << "null\n";
}

void benchmark(const std::vector<Key>& requests, bool byte_budget, std::size_t limit, std::size_t rounds,
               ReuseModel reuse, const PairwiseModels& models, const std::vector<Kind>& kinds) {
    std::array<std::size_t, policy_names.size()> capacities{};
    std::array<BenchmarkResult, policy_names.size()> results;
    std::array<CacheMemory, policy_names.size()> memory;
    const auto run = [&](std::size_t i, BenchmarkResult* result) {
        with_player(kinds[i], capacities[i], reuse, models,
                    [&](auto& player, std::size_t, std::size_t) { time_replay(player, requests, result); });
    };

    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const auto kind = kinds[i];
        capacities[i] = byte_budget ? capacity_for_budget(limit, [&](std::size_t capacity) {
            return prefill_memory(kind, capacity, reuse, models);
        }) : limit;
        run(i, nullptr);
    }
    for (std::size_t round = 0; round < rounds; ++round) {
        for (std::size_t offset = 0; offset < kinds.size(); ++offset) {
            const auto i = (round % kinds.size() + offset) % kinds.size();
            run(i, &results[i]);
        }
    }
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        AllocationTally tally;
        with_player<CountingAllocator>(kinds[i], capacities[i], reuse, models,
            [&](auto& player, std::size_t object_bytes, std::size_t model_bytes) {
                replay_keys(player, requests);
                check_benchmark_stats(player, results[i]);
                memory[i] = {object_bytes, tally.peak, model_bytes};
            }, &tally);
        if (tally.current || (byte_budget && memory[i].total() > limit)) {
            throw std::runtime_error("cache allocation accounting disagrees with the budget");
        }
    }

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "{\n  \"requests\": " << requests.size() << ",\n  \"benchmark_rounds\": " << rounds
              << ",\n  \"" << (byte_budget ? "budget_bytes" : "capacity_items") << "\": " << limit
              << ",\n  \"results\": [\n";
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        std::cout << (i ? ",\n" : "") << results[i].stats_json;
    }
    std::cout << "\n  ],\n  \"performance\": [\n";
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const auto wall = median(results[i].wall_seconds);
        std::cout << (i ? ",\n" : "") << "    {\"policy\": \"" << name(kinds[i])
                  << "\", \"capacity_items\": " << capacities[i]
                  << ", \"median_cpu_seconds\": " << median(results[i].cpu_seconds)
                  << ", \"median_wall_seconds\": " << wall << ", \"requests_per_second\": ";
        if (wall > 0) {
            std::cout << requests.size() / wall;
        } else {
            std::cout << "null";
        }
        std::cout << ", \"peak_accounted_bytes\": " << memory[i].total()
                  << ", \"model_bytes\": " << memory[i].model_bytes
                  << ", \"cpu_seconds_samples\": ";
        print_samples(results[i].cpu_seconds);
        std::cout << ", \"wall_seconds_samples\": ";
        print_samples(results[i].wall_seconds);
        std::cout << '}';
    }
    std::cout << "\n  ]\n}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: cachetriage_pairwise <trace> <pairwise-model> <reuse-model> <items|bytes|compact-items|compact-bytes|compact-scan> <limit> <rounds-or-target-misses>\n";
        return 1;
    }
    const std::string_view mode = argv[4];
    const bool scanning = mode == "compact-scan";
    const bool compact = mode == "compact-items" || mode == "compact-bytes";
    std::size_t limit = 0, rounds = 0;
    if ((!scanning && !compact && mode != "items" && mode != "bytes") || !parse_unsigned(argv[5], limit) || limit == 0 ||
        !parse_unsigned(argv[6], rounds) || (!scanning && rounds == 0)) {
        std::cerr << "Choose a benchmark mode with a positive limit and round count, or a scan with a miss target.\n";
        return 1;
    }
    try {
        const auto models = load_pairwise_models(argv[2]);
        const auto reuse = load_reuse_model(argv[3]);
        if (!reuse) {
            throw std::runtime_error("cannot read the eight-hit gate's reuse model");
        }
        std::vector<Key> requests;
        for_each_request(argv[1], [&](Key key) { requests.push_back(key); });
        if (scanning) {
            scan(requests, limit, rounds, *reuse, models);
        } else {
            const auto kinds = compact ? std::vector<Kind>{Kind::sieve, Kind::logistic}
                : std::vector<Kind>{Kind::sieve, Kind::eight_hit, Kind::history_rule, Kind::logistic, Kind::boosted};
            benchmark(requests, mode == "bytes" || mode == "compact-bytes", limit, rounds, *reuse, models, kinds);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
