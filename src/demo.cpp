#include "cache_budget.hpp"
#include "counting_allocator.hpp"
#include "pairwise_model.hpp"
#include "reuse_model.hpp"
#include "trace_reader.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace cachetriage;
using Profile = std::array<unsigned char, 256>;

class ProfileStore {
public:
    explicit ProfileStore(const std::vector<Key>& requests) {
        sqlite3* connection = nullptr;
        const int status = sqlite3_open("", &connection);  // Private temporary database.
        db_.reset(connection);
        check(status, SQLITE_OK, "open SQLite");

        auto keys = requests;
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        profiles_ = keys.size();
        execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)");
        execute("BEGIN");
        auto insert = prepare("INSERT INTO profiles (id, payload) VALUES (?1, ?2)");
        Profile payload;
        payload.fill(0xa5);
        for (Key key : keys) {
            std::memcpy(payload.data(), &key, sizeof(key));
            check(sqlite3_bind_int64(insert.get(), 1, database_key(key)), SQLITE_OK, "bind seed key");
            check(sqlite3_bind_blob(insert.get(), 2, payload.data(), static_cast<int>(payload.size()),
                                    SQLITE_TRANSIENT),
                  SQLITE_OK, "bind seed payload");
            check(sqlite3_step(insert.get()), SQLITE_DONE, "seed profile");
            check(sqlite3_reset(insert.get()), SQLITE_OK, "reset seed statement");
        }
        execute("COMMIT");
        lookup_ = prepare("SELECT payload FROM profiles WHERE id = ?1");
    }

    Profile fetch(Key key) {
        auto* statement = lookup_.get();
        check(sqlite3_bind_int64(statement, 1, database_key(key)), SQLITE_OK, "bind lookup key");
        check(sqlite3_step(statement), SQLITE_ROW, "read profile");
        ++calls_;

        Profile payload;
        const auto* blob = sqlite3_column_blob(statement, 0);
        if (!blob || sqlite3_column_bytes(statement, 0) != static_cast<int>(payload.size())) {
            throw std::runtime_error("unexpected profile payload");
        }
        std::memcpy(payload.data(), blob, payload.size());
        check(sqlite3_reset(statement), SQLITE_OK, "reset lookup");
        return payload;
    }

    std::uint64_t calls() const { return calls_; }
    std::size_t profiles() const { return profiles_; }

private:
    using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

    // Preserve all 64-bit keys in SQLite's signed integer primary key.
    static sqlite3_int64 database_key(Key key) {
        if (key <= static_cast<Key>(std::numeric_limits<sqlite3_int64>::max())) {
            return static_cast<sqlite3_int64>(key);
        }
        return -1 - static_cast<sqlite3_int64>(std::numeric_limits<Key>::max() - key);
    }

    void check(int status, int expected, const char* action) const {
        if (status != expected) {
            throw std::runtime_error(std::string(action) + ": " +
                                     (db_ ? sqlite3_errmsg(db_.get()) : sqlite3_errstr(status)));
        }
    }

    void execute(const char* sql) {
        check(sqlite3_exec(db_.get(), sql, nullptr, nullptr, nullptr), SQLITE_OK, sql);
    }

    Statement prepare(const char* sql) {
        sqlite3_stmt* statement = nullptr;
        const int status = sqlite3_prepare_v2(db_.get(), sql, -1, &statement, nullptr);
        Statement owned(statement, sqlite3_finalize);
        check(status, SQLITE_OK, "prepare statement");
        return owned;
    }

    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_{nullptr, sqlite3_close};
    Statement lookup_{nullptr, sqlite3_finalize};
    std::uint64_t calls_ = 0;
    std::size_t profiles_ = 0;
};

template <template <typename> class Allocator = std::allocator>
class PlainProfiles {
public:
    PlainProfiles(std::size_t capacity, Policy policy, Allocator<Profile> allocator = {})
        : cache_(capacity, policy, allocator) {}

    const Profile* find(Key key, std::uint64_t) { return cache_.find(key); }

    void insert(Key key, const Profile& payload, std::uint64_t) { cache_.insert(key, payload); }

    std::uint64_t model_calls() const { return 0; }

private:
    Cache<Profile, Allocator> cache_;
};

struct LearnedProfile {
    Profile payload;
    ReuseFeatures features;
};

template <typename Selector, template <typename> class Allocator = std::allocator>
class LearnedProfiles {
public:
    LearnedProfiles(std::size_t capacity, Selector selector, Allocator<LearnedProfile> allocator = {})
        : cache_(capacity, Policy::sieve, allocator), selector_(selector) {}

    const Profile* find(Key key, std::uint64_t request) {
        auto* resident = cache_.find(key);
        if (!resident) {
            return nullptr;
        }
        resident->features.record_hit(request);
        return &resident->payload;
    }

    void insert(Key key, const Profile& payload, std::uint64_t request) {
        const auto select = [&](const EvictionShortlist& shortlist, const auto& values) noexcept {
            return selector_.select(shortlist, request, [&](std::size_t i) noexcept -> const ReuseFeatures& {
                return values[i]->features;
            });
        };
        cache_.insert(key, {payload, {request, 0}}, nullptr, select);
    }

    std::uint64_t model_calls() const { return selector_.calls(); }

private:
    Cache<LearnedProfile, Allocator> cache_;
    Selector selector_;
};

struct Trial {
    std::uint64_t backend_calls = 0;
    std::uint64_t model_calls = 0;
    double mean_us = 0;
    double p50_us = 0;
    double p95_us = 0;
    double injected_us_per_fetch = 0;
    double cpu_seconds = 0;
};

using Clock = std::chrono::steady_clock;
using FetchDelay = std::chrono::microseconds;

Clock::duration wait_for_fetch_delay(FetchDelay delay) {
    // Busy waiting gives microsecond control without sleep timer granularity.
    // The experiment measures request latency, not CPU use under this delay.
    const auto start = Clock::now();
    const auto deadline = start + delay;
    auto now = start;
    do {
        now = Clock::now();
    } while (now < deadline);
    return now - start;
}

template <typename Cache>
Trial run_requests(Cache& cache, ProfileStore& store, const std::vector<Key>& requests,
                   FetchDelay delay) {
    Clock::duration injected{};
    std::vector<double> latencies;
    latencies.reserve(requests.size());
    const auto calls_before = store.calls();
    const auto cpu_start = std::clock();

    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto key = requests[i];
        Profile fetched;
        const auto start = Clock::now();
        const auto* payload = cache.find(key, i + 1);
        if (!payload) {
            fetched = store.fetch(key);
            if (delay.count() != 0) {
                injected += wait_for_fetch_delay(delay);
            }
            cache.insert(key, fetched, i + 1);
            payload = &fetched;
        }
        const auto end = Clock::now();

        Key returned_key = 0;
        std::memcpy(&returned_key, payload->data(), sizeof(returned_key));
        if (returned_key != key) {
            throw std::runtime_error("returned profile belongs to a different key");
        }
        latencies.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    Trial trial{store.calls() - calls_before, cache.model_calls()};
    trial.cpu_seconds = (std::clock() - cpu_start) / static_cast<double>(CLOCKS_PER_SEC);
    if (trial.backend_calls != 0) {
        trial.injected_us_per_fetch =
            std::chrono::duration<double, std::micro>(injected).count() / trial.backend_calls;
    }
    if (!latencies.empty()) {
        trial.mean_us = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        std::sort(latencies.begin(), latencies.end());
        trial.p50_us = latencies[latencies.size() - latencies.size() / 2 - 1];
        trial.p95_us = latencies[latencies.size() - latencies.size() / 20 - 1];
    }
    return trial;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

double median_latency(const std::vector<Trial>& trials, double Trial::*field) {
    std::vector<double> values;
    for (const auto& trial : trials) {
        values.push_back(trial.*field);
    }
    return median(std::move(values));
}

enum class DemoPolicy { lru, sieve, always_score, cachetriage };

Trial run_trial(DemoPolicy policy, std::size_t capacity, ReuseModel model, ReuseGate gate,
                ProfileStore& store, const std::vector<Key>& requests, FetchDelay delay = {}) {
    if (policy == DemoPolicy::lru || policy == DemoPolicy::sieve) {
        PlainProfiles<> cache(capacity, policy == DemoPolicy::lru ? Policy::lru : Policy::sieve);
        return run_requests(cache, store, requests, delay);
    }
    LearnedProfiles<ReuseSelector> cache(capacity, {model, policy == DemoPolicy::always_score ? ReuseGate{} : gate});
    return run_requests(cache, store, requests, delay);
}

void print_demo_header(const ProfileStore& store, std::size_t requests, std::size_t capacity,
                       ReuseGate gate, std::size_t rounds) {
    std::cout << std::setprecision(9) << "{\n  \"backend\": \"temporary_sqlite\","
              << "\n  \"sqlite_version\": \"" << sqlite3_libversion() << '"'
              << ",\n  \"capacity_items\": " << capacity
              << ",\n  \"requests\": " << requests
              << ",\n  \"profiles\": " << store.profiles()
              << ",\n  \"profile_bytes\": " << sizeof(Profile)
              << ",\n  \"" << gate.threshold_field() << "\": " << gate.threshold
              << ",\n  \"demo_rounds\": " << rounds;
}

void print_trial_summary(const char* name, const std::vector<Trial>& trials, bool has_reads,
                         bool report_delay = false, bool report_cpu = false) {
    const auto& first = trials.front();
    for (const auto& trial : trials) {
        if (trial.backend_calls != first.backend_calls || trial.model_calls != first.model_calls) {
            throw std::runtime_error("call counts changed between demo rounds");
        }
    }
    std::cout << "{\"policy\": \"" << name
              << "\", \"backend_calls\": " << first.backend_calls
              << ", \"model_calls\": " << first.model_calls
              << ", \"mean_latency_us\": ";
    if (has_reads) {
        std::cout << median_latency(trials, &Trial::mean_us)
                  << ", \"p50_latency_us\": " << median_latency(trials, &Trial::p50_us)
                  << ", \"p95_latency_us\": " << median_latency(trials, &Trial::p95_us);
    } else {
        std::cout << "null, \"p50_latency_us\": null, \"p95_latency_us\": null";
    }
    if (report_delay) {
        std::cout << ", \"median_injected_us_per_fetch\": ";
        if (first.backend_calls != 0) {
            std::cout << median_latency(trials, &Trial::injected_us_per_fetch);
        } else {
            std::cout << "null";
        }
    }
    if (report_cpu) {
        std::cout << ", \"median_cpu_seconds\": " << median_latency(trials, &Trial::cpu_seconds);
    }
    std::cout << '}';
}

void run_demo(const std::vector<Key>& requests, std::size_t capacity, ReuseModel model,
              ReuseGate gate, std::size_t rounds) {
    ProfileStore store(requests);
    constexpr std::array<const char*, 4> names{"lru", "sieve", "always_score", "cachetriage"};
    std::array<std::vector<Trial>, names.size()> results;
    const auto run = [&](std::size_t i) {
        return run_trial(static_cast<DemoPolicy>(i), capacity, model, gate, store, requests);
    };

    for (std::size_t i = 0; i < names.size(); ++i) {
        run(i);
    }
    for (std::size_t round = 0; round < rounds; ++round) {
        for (std::size_t offset = 0; offset < names.size(); ++offset) {
            const auto i = (round % names.size() + offset) % names.size();
            results[i].push_back(run(i));
        }
    }

    print_demo_header(store, requests.size(), capacity, gate, rounds);
    std::cout << ",\n  \"results\": [\n";
    for (std::size_t i = 0; i < names.size(); ++i) {
        std::cout << (i == 0 ? "" : ",\n") << "    ";
        print_trial_summary(names[i], results[i], !requests.empty());
    }
    std::cout << "\n  ]\n}\n";
}

struct DelayTrials {
    std::vector<Trial> sieve;
    std::vector<Trial> triage;
};

void print_paired_deltas(const DelayTrials& trials, const char* metric, double Trial::*field) {
    std::vector<double> deltas;
    for (std::size_t round = 0; round < trials.sieve.size(); ++round) {
        deltas.push_back((trials.triage[round].*field) - (trials.sieve[round].*field));
    }
    std::cout << ", \"median_paired_" << metric << "_delta_us\": " << median(deltas)
              << ", \"paired_" << metric << "_deltas_us\": [";
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << deltas[i];
    }
    std::cout << ']';
}

void run_delay_sweep(const std::vector<Key>& requests, std::size_t capacity, ReuseModel model,
                     ReuseGate gate, std::size_t rounds, const std::vector<FetchDelay>& delays) {
    if (requests.empty()) {
        throw std::runtime_error("delay sweep requires at least one read");
    }
    ProfileStore store(requests);
    std::vector<DelayTrials> results(delays.size());
    const auto run = [&](DemoPolicy policy, FetchDelay delay) {
        return run_trial(policy, capacity, model, gate, store, requests, delay);
    };

    run(DemoPolicy::sieve, {});
    run(DemoPolicy::cachetriage, {});
    for (std::size_t round = 0; round < rounds; ++round) {
        for (std::size_t offset = 0; offset < delays.size(); ++offset) {
            const auto i = (round % delays.size() + offset) % delays.size();
            auto& trials = results[i];
            if (round % 2 == 0) {
                trials.sieve.push_back(run(DemoPolicy::sieve, delays[i]));
                trials.triage.push_back(run(DemoPolicy::cachetriage, delays[i]));
            } else {
                trials.triage.push_back(run(DemoPolicy::cachetriage, delays[i]));
                trials.sieve.push_back(run(DemoPolicy::sieve, delays[i]));
            }
        }
    }

    print_demo_header(store, requests.size(), capacity, gate, rounds);
    std::cout << ",\n  \"delay_sweep\": [\n";
    for (std::size_t i = 0; i < delays.size(); ++i) {
        const auto& trials = results[i];
        std::cout << (i == 0 ? "" : ",\n") << "    {\"added_fetch_delay_us\": " << delays[i].count();
        print_paired_deltas(trials, "mean", &Trial::mean_us);
        print_paired_deltas(trials, "p95", &Trial::p95_us);
        std::cout << ", \"results\": [";
        print_trial_summary("sieve", trials.sieve, true, true);
        std::cout << ", ";
        print_trial_summary("cachetriage", trials.triage, true, true);
        std::cout << "]}";
    }
    std::cout << "\n  ]\n}\n";
}

template <typename Use>
void with_counted_profiles(bool logistic, std::size_t capacity, const LinearOverride& model,
                           AllocationTally& tally, Use use) {
    if (logistic) {
        LearnedProfiles<PairwiseSelector, CountingAllocator> cache(capacity, PairwiseSelector(model),
                                                                   CountingAllocator<LearnedProfile>(tally));
        use(cache, sizeof(LearnedProfiles<PairwiseSelector>), LinearOverride::bytes());
    } else {
        PlainProfiles<CountingAllocator> cache(capacity, Policy::sieve, CountingAllocator<Profile>(tally));
        use(cache, sizeof(PlainProfiles<>), 0);
    }
}

void run_pairwise_budget(const std::vector<Key>& requests, std::size_t budget,
                         const LinearOverride& model, std::size_t rounds) {
    ProfileStore store(requests);
    constexpr std::array<const char*, 2> names{"sieve", "pairwise_logistic"};
    std::array<std::size_t, names.size()> capacities{};
    std::array<CacheMemory, names.size()> memory{};
    std::array<std::vector<Trial>, names.size()> trials;
    const auto run = [&](std::size_t i) {
        if (i == 0) {
            PlainProfiles<> cache(capacities[i], Policy::sieve);
            return run_requests(cache, store, requests, {});
        }
        LearnedProfiles<PairwiseSelector> cache(capacities[i], PairwiseSelector(model));
        return run_requests(cache, store, requests, {});
    };
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto prefill = [&](std::size_t capacity) {
            AllocationTally tally;
            CacheMemory result;
            with_counted_profiles(i != 0, capacity, model, tally,
                [&](auto& cache, std::size_t object_bytes, std::size_t model_bytes) {
                    const Profile payload{};
                    for (std::size_t key = 0; key < capacity; ++key) {
                        cache.insert(key, payload, key + 1);
                    }
                    result = {object_bytes, tally.peak, model_bytes};
                });
            return result;
        };
        capacities[i] = capacity_for_budget(budget, prefill);
        run(i);
    }

    for (std::size_t round = 0; round < rounds; ++round) {
        const auto first = round % names.size();
        trials[first].push_back(run(first));
        trials[1 - first].push_back(run(1 - first));
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
        AllocationTally tally;
        with_counted_profiles(i != 0, capacities[i], model, tally,
            [&](auto& cache, std::size_t object_bytes, std::size_t model_bytes) {
                const auto check = run_requests(cache, store, requests, {});
                if (check.backend_calls != trials[i].front().backend_calls ||
                    check.model_calls != trials[i].front().model_calls) {
                    throw std::runtime_error("counted and timed demo passes disagree");
                }
                memory[i] = {object_bytes, tally.peak, model_bytes};
            });
        if (tally.current || memory[i].total() > budget) {
            throw std::runtime_error("demo cache exceeded its accounted budget");
        }
    }

    std::cout << std::setprecision(9) << "{\n  \"backend\": \"temporary_sqlite\","
              << "\n  \"sqlite_version\": \"" << sqlite3_libversion() << '"'
              << ",\n  \"budget_bytes\": " << budget
              << ",\n  \"requests\": " << requests.size()
              << ",\n  \"profiles\": " << store.profiles()
              << ",\n  \"profile_bytes\": " << sizeof(Profile)
              << ",\n  \"demo_rounds\": " << rounds
              << ",\n  \"results\": [\n";
    for (std::size_t i = 0; i < names.size(); ++i) {
        std::cout << (i == 0 ? "" : ",\n") << "    ";
        print_trial_summary(names[i], trials[i], !requests.empty(), false, true);
    }
    std::cout << "\n  ],\n  \"memory\": [\n";
    for (std::size_t i = 0; i < names.size(); ++i) {
        std::cout << (i == 0 ? "" : ",\n") << "    {\"policy\": \"" << names[i]
                  << "\", \"capacity_items\": " << capacities[i]
                  << ", \"peak_accounted_bytes\": " << memory[i].total()
                  << ", \"model_bytes\": " << memory[i].model_bytes << '}';
    }
    std::cout << "\n  ]\n}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string_view(argv[1]) == "--pairwise-budget") {
        std::size_t budget = 0, rounds = 0;
        if (argc != 6 || !parse_unsigned(argv[2], budget) || budget == 0 ||
            !parse_unsigned(argv[5], rounds) || rounds == 0) {
            std::cerr << "Usage: cachetriage_demo --pairwise-budget <bytes> <trace-file> <pairwise-model> <rounds>\n";
            return 1;
        }
        try {
            const auto model = load_pairwise_models(argv[4]).logistic;
            std::vector<Key> requests;
            for_each_request(argv[3], [&](Key key) { requests.push_back(key); });
            run_pairwise_budget(requests, budget, model, rounds);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }

    const bool sweep_mode = argc >= 8 && std::string_view(argv[6]) == "--delay-sweep";
    const bool hit_gate = argc == 7 && std::string_view(argv[6]) == "--hit-gate";
    if (argc != 6 && !sweep_mode && !hit_gate) {
        std::cerr << "Usage: cachetriage_demo <capacity-items> <trace-file> <model-file> <gate-threshold> "
                     "<rounds> [--hit-gate | --delay-sweep <microseconds> ...]\n";
        return 1;
    }

    std::size_t capacity = 0, rounds = 0;
    ReuseGate gate{hit_gate ? GateRule::victim_hits : GateRule::age_gap, 0};
    if (!parse_unsigned(argv[1], capacity) || !parse_unsigned(argv[4], gate.threshold) ||
        !parse_unsigned(argv[5], rounds) || rounds == 0 || (hit_gate && gate.threshold == 0)) {
        std::cerr << "Capacity and age gap must be nonnegative. Rounds and minimum victim hits must be positive.\n";
        return 1;
    }

    std::vector<FetchDelay> delays;
    if (sweep_mode) {
        for (int i = 7; i < argc; ++i) {
            std::uint32_t delay = 0;
            if (!parse_unsigned(argv[i], delay)) {
                std::cerr << "Fetch delay must be a nonnegative number of microseconds.\n";
                return 1;
            }
            delays.emplace_back(delay);
        }
    }

    try {
        const auto model = load_reuse_model(argv[3]);
        if (!model) {
            throw std::runtime_error(std::string("Cannot read a valid reuse model from ") + argv[3]);
        }
        std::vector<Key> requests;
        for_each_request(argv[2], [&](Key key) { requests.push_back(key); });
        if (sweep_mode) {
            run_delay_sweep(requests, capacity, *model, gate, rounds, delays);
        } else {
            run_demo(requests, capacity, *model, gate, rounds);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
