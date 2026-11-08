#include "cachetriage/cache.hpp"
#include "counting_allocator.hpp"
#include "reuse_model.hpp"
#include "replay_support.hpp"
#include "trace_reader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace {

using cachetriage::AllocationTally;
using cachetriage::Cache;
using cachetriage::CountingAllocator;
using cachetriage::EvictionShortlist;
using cachetriage::GateRule;
using cachetriage::Key;
using cachetriage::Policy;
using cachetriage::ReuseFeatures;
using cachetriage::ReuseGate;
using cachetriage::ReuseModel;
using cachetriage::ReuseSelector;
using cachetriage::for_each_request;
using cachetriage::load_reuse_model;
using cachetriage::parse_unsigned;

using namespace cachetriage::replay;

std::uint64_t run_oracle(const std::vector<Key>& requests, std::size_t lookahead, Replay<>& oracle) {
    std::unordered_map<Key, std::vector<std::size_t>> positions;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        positions[requests[i]].push_back(i);
    }

    std::uint64_t swaps = 0;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        if (requests.size() - i - 1 < lookahead) {
            oracle.read(requests[i]);
            continue;
        }
        const auto reused = [&](Key key) noexcept {
            // Every resident came from this request stream.
            const auto& occurrences = positions.find(key)->second;
            const auto next = std::upper_bound(occurrences.begin(), occurrences.end(), i);
            return next != occurrences.end() && *next <= i + lookahead;
        };
        const auto select = [&](const EvictionShortlist& shortlist) noexcept -> std::size_t {
            if (reused(shortlist.keys[0])) {
                for (std::size_t candidate = 1; candidate < shortlist.size; ++candidate) {
                    if (!reused(shortlist.keys[candidate])) {
                        ++swaps;
                        return candidate;
                    }
                }
            }
            return 0;
        };
        oracle.read(requests[i], nullptr, select);
    }
    return swaps;
}

void run_benchmark(const std::vector<Key>& requests, std::size_t capacity, ReuseModel model,
                   ReuseGate gate, std::size_t rounds) {
    constexpr std::array<const char*, 4> names{"lru", "sieve", "always_score", "cachetriage"};
    std::array<BenchmarkResult, names.size()> results;
    const auto baseline_policy = [](std::size_t i) { return i == 0 ? Policy::lru : Policy::sieve; };
    const auto gate_for = [=](std::size_t i) { return i == 2 ? ReuseGate{} : gate; };
    const auto run = [&](std::size_t i, BenchmarkResult* result) {
        if (i < 2) {
            Replay<> player(names[i], capacity, baseline_policy(i));
            time_replay(player, requests, result);
        } else {
            LearnedReplay<> player(capacity, model, gate_for(i));
            time_replay(player, requests, result);
        }
    };

    for (std::size_t i = 0; i < names.size(); ++i) {
        run(i, nullptr);  // Warm up each policy, then start all measured runs empty.
    }
    for (std::size_t round = 0; round < rounds; ++round) {
        for (std::size_t offset = 0; offset < names.size(); ++offset) {
            const auto i = (round % names.size() + offset) % names.size();
            run(i, &results[i]);
        }
    }

    for (std::size_t i = 0; i < names.size(); ++i) {
        auto& result = results[i];
        AllocationTally tally;
        if (i < 2) {
            Replay<CountingAllocator> player(names[i], capacity, baseline_policy(i),
                                             CountingAllocator<std::monostate>(tally));
            replay_keys(player, requests);
            check_benchmark_stats(player, result);
            result.replay_inline_bytes = sizeof(Replay<>);
        } else {
            LearnedReplay<CountingAllocator> player(capacity, model, gate_for(i),
                                                    CountingAllocator<ReuseFeatures>(tally));
            replay_keys(player, requests);
            check_benchmark_stats(player, result);
            result.replay_inline_bytes = sizeof(LearnedReplay<>);
            result.feature_bytes_per_resident = sizeof(ReuseFeatures);
            result.shared_model_table_bytes = ReuseModel::shared_table_bytes();
        }
        if (tally.current != 0) {
            throw std::runtime_error("unbalanced cache allocation count");
        }
        result.peak_requested_heap_bytes = tally.peak;
    }

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "{\n  \"capacity_items\": " << capacity
              << ",\n  \"requests\": " << requests.size()
              << ",\n  \"" << gate.threshold_field() << "\": " << gate.threshold
              << ",\n  \"benchmark_rounds\": " << rounds << ",\n  \"results\": [\n";
    for (std::size_t i = 0; i < names.size(); ++i) {
        std::cout << (i == 0 ? "" : ",\n") << results[i].stats_json;
    }
    std::cout << "\n  ],\n  \"performance\": [\n";
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto& result = results[i];
        const auto wall = median(result.wall_seconds);
        std::cout << (i == 0 ? "" : ",\n") << "    {\"policy\": \"" << names[i]
                  << "\", \"median_cpu_seconds\": " << median(result.cpu_seconds)
                  << ", \"median_wall_seconds\": " << wall << ", \"requests_per_second\": ";
        if (wall > 0) {
            std::cout << requests.size() / wall;
        } else {
            std::cout << "null";
        }
        std::cout << ", \"replay_inline_bytes\": " << result.replay_inline_bytes
                  << ", \"peak_requested_heap_bytes\": " << result.peak_requested_heap_bytes
                  << ", \"peak_accounted_bytes\": "
                  << result.replay_inline_bytes + result.peak_requested_heap_bytes + result.shared_model_table_bytes
                  << ", \"shared_model_table_bytes\": " << result.shared_model_table_bytes
                  << ", \"feature_bytes_per_resident\": " << result.feature_bytes_per_resident
                  << ", \"cpu_seconds_samples\": ";
        print_samples(result.cpu_seconds);
        std::cout << ", \"wall_seconds_samples\": ";
        print_samples(result.wall_seconds);
        std::cout << '}';
    }
    std::cout << "\n  ]\n}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    constexpr auto usage =
        "Usage: cachetriage <capacity-items> <trace-file> "
        "[--sieve-decisions | --sieve-oracle <reads> | --always-score <model-file> "
        "| --triage <model-file> <max-age-gap> "
        "| --triage-hits <model-file> <min-victim-hits> "
        "| --benchmark <model-file> <max-age-gap> <rounds> "
        "| --benchmark-hits <model-file> <min-victim-hits> <rounds>]\n";
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage;
        return 0;
    }
    const bool decisions_only = argc == 4 && std::string_view(argv[3]) == "--sieve-decisions";
    const bool oracle_mode = argc == 5 && std::string_view(argv[3]) == "--sieve-oracle";
    const bool always_score_mode = argc == 5 && std::string_view(argv[3]) == "--always-score";
    const std::string_view mode = argc > 3 ? argv[3] : "";
    const bool triage_mode = argc == 6 && (mode == "--triage" || mode == "--triage-hits");
    const bool benchmark_mode = argc == 7 && (mode == "--benchmark" || mode == "--benchmark-hits");
    const bool hit_gate = mode == "--triage-hits" || mode == "--benchmark-hits";
    if (argc != 3 && !decisions_only && !oracle_mode && !always_score_mode && !triage_mode &&
        !benchmark_mode) {
        std::cerr << usage;
        return 1;
    }

    std::size_t capacity = 0;
    if (!parse_unsigned(argv[1], capacity)) {
        std::cerr << "Capacity must be a nonnegative integer that fits in size_t.\n";
        return 1;
    }
    std::size_t lookahead = 0;
    if (oracle_mode && (!parse_unsigned(argv[4], lookahead) || lookahead == 0)) {
        std::cerr << "Oracle lookahead must be a positive number of reads.\n";
        return 1;
    }
    ReuseGate gate;
    if (triage_mode || benchmark_mode) {
        gate.rule = hit_gate ? GateRule::victim_hits : GateRule::age_gap;
        if (!parse_unsigned(argv[5], gate.threshold) || (hit_gate && gate.threshold == 0)) {
            std::cerr << (hit_gate ? "Minimum victim hits must be positive.\n"
                                   : "Gate age gap must be a nonnegative number of reads.\n");
            return 1;
        }
    }

    std::size_t rounds = 0;
    if (benchmark_mode && (!parse_unsigned(argv[6], rounds) || rounds == 0)) {
        std::cerr << "Benchmark rounds must be positive.\n";
        return 1;
    }

    std::optional<ReuseModel> model;
    if (always_score_mode || triage_mode || benchmark_mode) {
        model = load_reuse_model(argv[4]);
        if (!model) {
            std::cerr << "Cannot read a valid reuse model from " << argv[4] << '\n';
            return 1;
        }
    }
    std::optional<LearnedReplay<>> learned;
    if (always_score_mode || triage_mode) {
        learned.emplace(capacity, *model, gate);
    }
    Replay<> lru("lru", capacity, Policy::lru);
    Replay<> sieve("sieve", capacity, Policy::sieve);
    std::vector<Key> recorded_requests;
    try {
        for_each_request(argv[2], [&](Key key) {
            if (benchmark_mode) {
                recorded_requests.push_back(key);
            } else if (decisions_only) {
                EvictionShortlist shortlist;
                if (sieve.read(key, &shortlist)) {
                    std::cout << sieve.stats.requests();
                    for (std::size_t i = 0; i < shortlist.size; ++i) {
                        std::cout << ',' << shortlist.keys[i];
                    }
                    std::cout << '\n';
                }
            } else {
                lru.read(key);
                sieve.read(key);
                if (oracle_mode) {
                    recorded_requests.push_back(key);
                }
                if (learned) {
                    learned->read(key);
                }
            }
        });
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    if (benchmark_mode) {
        try {
            run_benchmark(recorded_requests, capacity, *model, gate, rounds);
        } catch (const std::exception& error) {
            std::cerr << "Benchmark failed: " << error.what() << '\n';
            return 1;
        }
        return 0;
    }

    if (decisions_only) {
        std::cout.flush();
        return std::cout ? 0 : 1;
    }

    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "{\n  \"capacity_items\": " << capacity
              << ",\n  \"requests\": " << lru.stats.requests();
    if (oracle_mode) {
        std::cout << ",\n  \"oracle_lookahead_reads\": " << lookahead;
    }
    if (gate.rule != GateRule::always) {
        std::cout << ",\n  \"" << gate.threshold_field() << "\": " << gate.threshold;
    }
    std::cout << ",\n  \"results\": [\n";
    lru.stats.print_json(std::cout);
    std::cout << ",\n";
    sieve.stats.print_json(std::cout);
    if (oracle_mode) {
        Replay<> oracle("sieve_oracle", capacity, Policy::sieve);
        const auto swaps = run_oracle(recorded_requests, lookahead, oracle);
        std::cout << ",\n";
        oracle.stats.print_json(std::cout, swaps);
    }
    if (learned) {
        std::cout << ",\n";
        learned->print_json(std::cout);
    }
    std::cout << "\n  ]\n}\n";
    return 0;
}
