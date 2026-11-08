#pragma once

#include "cachetriage/cache.hpp"

#include <cstddef>
#include <stdexcept>

namespace cachetriage {

struct CacheMemory {
    std::size_t inline_bytes = 0;
    std::size_t heap_bytes = 0;
    std::size_t model_bytes = 0;

    std::size_t total() const { return inline_bytes + heap_bytes + model_bytes; }
};

template <typename Measure>
std::size_t capacity_for_budget(std::size_t budget, Measure measure) {
    if (measure(0).total() > budget) {
        throw std::runtime_error("memory budget is smaller than fixed policy state");
    }
    std::size_t low = 0;
    std::size_t high = budget / sizeof(Key) + 1;
    // Unique inserts grow the cache monotonically. Full replacement reuses both nodes.
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (measure(middle).total() <= budget) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

}  // namespace cachetriage
