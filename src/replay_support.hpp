#pragma once

#include "cachetriage/cache.hpp"
#include "counting_allocator.hpp"
#include "reuse_model.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace cachetriage::replay {

struct ReplayStats {
    const char* name;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;

    std::uint64_t requests() const { return hits + misses; }

    void print_json(std::ostream& out, std::optional<std::uint64_t> swaps = std::nullopt,
                    std::optional<std::uint64_t> model_calls = std::nullopt) const {
        out << "    {\"policy\": \"" << name << "\", \"hits\": " << hits
            << ", \"misses\": " << misses << ", \"evictions\": " << evictions
            << ", \"miss_rate\": ";
        if (requests() == 0) {
            out << "null";
        } else {
            out << static_cast<double>(misses) / requests();
        }
        if (swaps) {
            out << ", \"victim_swaps\": " << *swaps;
        }
        if (model_calls) {
            out << ", \"model_calls\": " << *model_calls << ", \"model_calls_per_eviction\": ";
            if (evictions == 0) {
                out << "null";
            } else {
                out << static_cast<double>(*model_calls) / evictions;
            }
        }
        out << '}';
    }
};

template <template <typename> class Allocator = std::allocator>
struct Replay {
    ReplayStats stats;
    Cache<std::monostate, Allocator> cache;

    Replay(const char* name, std::size_t capacity, Policy policy,
           Allocator<std::monostate> allocator = {})
        : stats{name}, cache(capacity, policy, allocator) {}

    void print_json(std::ostream& out) const { stats.print_json(out); }

    template <typename Select = std::nullptr_t>
    bool read(Key key, EvictionShortlist* shortlist = nullptr, Select select = nullptr) {
        if (cache.find(key)) {
            ++stats.hits;
            return false;
        }

        ++stats.misses;
        const bool evicted = cache.insert(key, {}, shortlist, select).has_value();
        stats.evictions += evicted;
        return evicted;
    }
};

template <template <typename> class Allocator = std::allocator>
class LearnedReplay {
public:
    LearnedReplay(std::size_t capacity, ReuseModel model, ReuseGate gate,
                  Allocator<ReuseFeatures> allocator = {}, const char* name = nullptr)
        : stats_{name ? name : (gate.rule == GateRule::always ? "always_score" : "cachetriage")},
          cache_(capacity, Policy::sieve, allocator), selector_(model, gate) {}

    void read(Key key) {
        const auto request = stats_.requests() + 1;
        if (auto* features = cache_.find(key)) {
            ++stats_.hits;
            features->record_hit(request);
            return;
        }

        ++stats_.misses;
        const auto select = [&](const EvictionShortlist& shortlist, const auto& values) noexcept {
            return selector_.select(shortlist, request, [&](std::size_t i) noexcept -> const ReuseFeatures& {
                return *values[i];
            });
        };
        stats_.evictions += cache_.insert(key, {request, 0}, nullptr, select).has_value();
    }

    void print_json(std::ostream& out) const {
        stats_.print_json(out, selector_.swaps(), selector_.calls());
    }

private:
    ReplayStats stats_;
    Cache<ReuseFeatures, Allocator> cache_;
    ReuseSelector selector_;
};

struct BenchmarkResult {
    std::string stats_json;
    std::vector<double> cpu_seconds;
    std::vector<double> wall_seconds;
    std::size_t replay_inline_bytes = 0;
    std::size_t peak_requested_heap_bytes = 0;
    std::size_t feature_bytes_per_resident = 0;
    std::size_t shared_model_table_bytes = 0;
};

template <typename Player>
void replay_keys(Player& player, const std::vector<Key>& requests) {
    for (const Key key : requests) {
        player.read(key);
    }
}

template <typename Player>
void check_benchmark_stats(const Player& player, BenchmarkResult& result) {
    std::ostringstream out;
    out.precision(std::numeric_limits<double>::max_digits10);
    player.print_json(out);
    if (result.stats_json.empty()) {
        result.stats_json = out.str();
    } else if (result.stats_json != out.str()) {
        throw std::runtime_error("replay results changed between benchmark passes");
    }
}

template <typename Player>
void time_replay(Player& player, const std::vector<Key>& requests, BenchmarkResult* result) {
    if (!result) {
        replay_keys(player, requests);
        return;
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const auto cpu_start = std::clock();
    replay_keys(player, requests);
    const auto cpu_end = std::clock();
    const auto wall_end = std::chrono::steady_clock::now();
    if (cpu_start == static_cast<std::clock_t>(-1) || cpu_end == static_cast<std::clock_t>(-1) ||
        cpu_end < cpu_start) {
        throw std::runtime_error("process CPU time unavailable");
    }

    result->cpu_seconds.push_back(static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC);
    result->wall_seconds.push_back(std::chrono::duration<double>(wall_end - wall_start).count());
    check_benchmark_stats(player, *result);
}

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

inline void print_samples(const std::vector<double>& samples) {
    std::cout << '[';
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << samples[i];
    }
    std::cout << ']';
}

}  // namespace cachetriage::replay
