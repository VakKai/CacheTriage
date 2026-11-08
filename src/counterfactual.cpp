#include "cachetriage/cache.hpp"
#include "trace_reader.hpp"
#include "eviction_history.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using cachetriage::Cache;
using cachetriage::EvictionShortlist;
using cachetriage::Key;
using cachetriage::Policy;
using cachetriage::for_each_request;
using cachetriage::parse_unsigned;

using History = cachetriage::EvictionHistory;

using Sieve = Cache<History>;

template <typename OnEviction>
void walk_sieve(const std::vector<Key>& requests, std::size_t capacity, std::size_t horizon,
                OnEviction on_eviction, std::vector<std::size_t>* miss_prefix = nullptr) {
    Sieve cache(capacity, Policy::sieve);
    std::size_t misses = 0;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto key = requests[i];
        if (auto* resident = cache.find(key)) {
            resident->record_hit(i + 1);
        } else {
            ++misses;
            if (cache.size() == capacity && requests.size() - i - 1 >= horizon) {
                on_eviction(cache, i);
            }
            cache.insert(key, {i + 1, i + 1});
        }
        if (miss_prefix) {
            miss_prefix->push_back(misses);
        }
    }
}

std::vector<std::size_t> select_samples(const std::vector<std::size_t>& eligible, std::size_t limit) {
    const auto count = std::min(eligible.size(), limit);
    if (count == 0) {
        return {};
    }
    std::vector<std::size_t> samples;
    samples.reserve(count);
    const auto width = eligible.size() / count;
    const auto extra = eligible.size() % count;
    std::size_t start = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto bin = width + (i < extra);
        samples.push_back(eligible[start + bin / 2]);
        start += bin;
    }
    return samples;
}

std::size_t future_misses(Sieve& branch, const std::vector<Key>& requests,
                          std::size_t eviction, std::size_t horizon) {
    std::size_t misses = 0;
    for (std::size_t step = 1; step <= horizon; ++step) {
        const auto i = eviction + step;
        const auto key = requests[i];
        if (!branch.find(key)) {
            ++misses;
            branch.insert(key, {i + 1, i + 1});
        }
    }
    return misses;
}

void print_candidate(std::size_t rank, Key key, const History& features, std::size_t request,
                     std::size_t misses, std::size_t baseline_misses) {
    std::cout << "{\"rank\": " << rank << ", \"key\": " << key
              << ", \"age_reads\": " << request - features.last
              << ", \"resident_hits\": " << features.hits
              << ", \"residency_reads\": " << request - features.admitted
              << ", \"previous_gap_reads\": ";
    if (features.previous == 0) {
        std::cout << "null";
    } else {
        std::cout << features.last - features.previous;
    }
    std::cout << ", \"future_misses\": " << misses << ", \"misses_saved\": "
              << static_cast<std::int64_t>(baseline_misses) - static_cast<std::int64_t>(misses) << '}';
}

void evaluate_eviction(const Sieve& state, const std::vector<Key>& requests, std::size_t i,
                       std::size_t horizon, std::size_t observed_baseline) {
    const auto incoming = requests[i];
    const auto request = i + 1;
    EvictionShortlist shortlist;
    auto baseline = state.fork_for_replay();
    baseline->insert(incoming, {request, request}, &shortlist);
    const auto baseline_misses = future_misses(*baseline, requests, i, horizon);
    if (baseline_misses != observed_baseline) {
        throw std::runtime_error("forked baseline disagrees with uninterrupted SIEVE replay");
    }

    std::cout << "    {\"request\": " << request << ", \"incoming_key\": " << incoming
              << ", \"candidates\": [";
    print_candidate(0, shortlist.keys[0], *state.peek(shortlist.keys[0]), request,
                    baseline_misses, baseline_misses);
    for (std::size_t rank = 1; rank < shortlist.size; ++rank) {
        auto alternative = state.fork_for_replay();
        const auto select = [rank](const EvictionShortlist&) noexcept { return rank; };
        const auto removed = alternative->insert(incoming, {request, request}, nullptr, select);
        if (removed != shortlist.keys[rank]) {
            throw std::runtime_error("forked shortlist disagrees with baseline shortlist");
        }
        const auto misses = future_misses(*alternative, requests, i, horizon);
        std::cout << ", ";
        print_candidate(rank, *removed, *state.peek(*removed), request, misses, baseline_misses);
    }
    std::cout << "]}";
}

void run(const std::vector<Key>& requests, std::size_t capacity, std::size_t horizon,
         std::size_t limit) {
    std::vector<std::size_t> eligible;
    std::vector<std::size_t> miss_prefix{0};
    walk_sieve(requests, capacity, horizon,
               [&](const Sieve&, std::size_t i) { eligible.push_back(i); }, &miss_prefix);
    const auto samples = select_samples(eligible, limit);

    std::cout << "{\n  \"capacity_items\": " << capacity << ",\n  \"requests\": " << requests.size()
              << ",\n  \"horizon_reads\": " << horizon
              << ",\n  \"eligible_evictions\": " << eligible.size()
              << ",\n  \"sampled_decisions\": [";
    if (!samples.empty()) {
        std::size_t next = 0;
        walk_sieve(requests, capacity, horizon, [&](const Sieve& state, std::size_t i) {
            if (next < samples.size() && i == samples[next]) {
                std::cout << (next == 0 ? "\n" : ",\n");
                evaluate_eviction(state, requests, i, horizon,
                                  miss_prefix[i + horizon + 1] - miss_prefix[i + 1]);
                ++next;
            }
        });
        if (next != samples.size()) {
            throw std::runtime_error("could not replay all sampled decisions");
        }
        std::cout << '\n' << "  ";
    }
    std::cout << "]\n}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: cachetriage_counterfactual <capacity-items> <trace-file> <horizon-reads> <sample-limit>\n";
        return 1;
    }
    std::size_t capacity = 0, horizon = 0, limit = 0;
    if (!parse_unsigned(argv[1], capacity) || capacity < 2 ||
        !parse_unsigned(argv[3], horizon) || horizon == 0 ||
        !parse_unsigned(argv[4], limit) || limit == 0) {
        std::cerr << "Capacity must be at least two. Horizon and sample limit must be positive.\n";
        return 1;
    }
    try {
        std::vector<Key> requests;
        for_each_request(argv[2], [&](Key key) { requests.push_back(key); });
        run(requests, capacity, horizon, limit);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
