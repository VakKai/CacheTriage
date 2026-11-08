#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>

namespace cachetriage {

struct AllocationTally {
    std::size_t current = 0;
    std::size_t peak = 0;
};

// Counts bytes requested through the containers, including their rebound node
// and bucket allocators. Used only in the separate memory pass.
template <typename T>
class CountingAllocator {
public:
    using value_type = T;

    explicit CountingAllocator(AllocationTally& tally) noexcept : tally_(&tally) {}

    template <typename U>
    CountingAllocator(const CountingAllocator<U>& other) noexcept : tally_(other.tally_) {}

    T* allocate(std::size_t count) {
        auto* memory = std::allocator<T>{}.allocate(count);
        tally_->current += count * sizeof(T);
        tally_->peak = std::max(tally_->peak, tally_->current);
        return memory;
    }

    void deallocate(T* memory, std::size_t count) noexcept {
        tally_->current -= count * sizeof(T);
        std::allocator<T>{}.deallocate(memory, count);
    }

    template <typename U>
    bool operator==(const CountingAllocator<U>& other) const noexcept {
        return tally_ == other.tally_;
    }

    template <typename U>
    bool operator!=(const CountingAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

private:
    template <typename> friend class CountingAllocator;
    AllocationTally* tally_;
};

}  // namespace cachetriage
