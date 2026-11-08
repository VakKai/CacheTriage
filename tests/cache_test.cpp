#include "cachetriage/cache.hpp"

#include <cstdlib>
#include <iostream>
#include <new>
#include <string>

using cachetriage::Cache;
using cachetriage::Key;
using cachetriage::Policy;

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void expect_hit(Cache<Key>& cache, Key key) {
    const auto* value = cache.find(key);
    check(value && *value == key, "expected a hit with the stored value");
}

void expect_miss(Cache<Key>& cache, Key key, std::optional<Key> victim = std::nullopt) {
    check(cache.find(key) == nullptr, "expected a miss");
    check(cache.insert(key, key) == victim, "unexpected eviction victim");
}

void check_example() {
    Cache<Key> lru(3, Policy::lru);
    Cache<Key> sieve(3, Policy::sieve);
    for (auto* cache : {&lru, &sieve}) {
        expect_miss(*cache, 1);
        expect_miss(*cache, 2);
        expect_miss(*cache, 3);
        expect_hit(*cache, 3);
        expect_hit(*cache, 1);
        expect_miss(*cache, 4, 2);
    }
    expect_miss(lru, 5, 3);
    expect_miss(lru, 3, 1);
    expect_miss(sieve, 5, 4);
    expect_hit(sieve, 3);
    check(lru.size() == 3 && sieve.size() == 3, "evictions must keep capacity bounded");
}

void check_sieve_wrap() {
    Cache<Key> cache(3, Policy::sieve);
    for (Key key : {1, 2, 3}) {
        expect_miss(cache, key);
        expect_hit(cache, key);
    }
    // All three entries get a second chance. The scan wraps and evicts 1.
    expect_miss(cache, 4, 1);
    expect_miss(cache, 5, 2);
    expect_miss(cache, 6, 3);
}

void check_small_capacities(Policy policy) {
    Cache<Key> disabled(0, policy);
    expect_miss(disabled, 1);
    expect_miss(disabled, 1);
    check(disabled.size() == 0, "zero capacity must bypass admission");

    Cache<Key> single(1, policy);
    expect_miss(single, 1);
    expect_hit(single, 1);
    expect_miss(single, 2, 1);
    expect_miss(single, 3, 2);
    check(single.size() == 1, "capacity one must retain exactly one entry");
}

void check_update(Policy policy) {
    Cache<std::string> cache(2, policy);
    cache.insert(1, "old");
    cache.insert(2, "second");
    check(!cache.insert(1, "new"), "replacing a value must not evict");
    check(cache.size() == 2, "replacing a value must not change the size");
    check(cache.insert(3, "third") == 2, "replacing a value must touch the entry");
    const auto* value = cache.find(1);
    check(value && *value == "new", "expected the replacement value");
}

// Shared by the list and hash allocators after rebinding to node types.
template <typename T>
struct TestAllocator {
    using value_type = T;

    explicit TestAllocator(bool& blocked) noexcept : blocked(&blocked) {}
    template <typename U>
    TestAllocator(const TestAllocator<U>& other) noexcept : blocked(other.blocked) {}

    T* allocate(std::size_t count) {
        if (*blocked) {
            throw std::bad_alloc();
        }
        return std::allocator<T>{}.allocate(count);
    }

    void deallocate(T* memory, std::size_t count) noexcept {
        std::allocator<T>{}.deallocate(memory, count);
    }

    template <typename U>
    bool operator==(const TestAllocator<U>& other) const noexcept { return blocked == other.blocked; }
    template <typename U>
    bool operator!=(const TestAllocator<U>& other) const noexcept { return !(*this == other); }

    bool* blocked;
};

void check_recycled_nodes() {
    bool blocked = false;
    Cache<Key, TestAllocator> cache(2, Policy::sieve, TestAllocator<Key>(blocked));
    cache.insert(1, 10);
    cache.insert(2, 20);

    blocked = true;
    const auto select_second = [](const cachetriage::EvictionShortlist&, const auto& values) noexcept {
        return *values[1] == 20 ? std::size_t{1} : std::size_t{0};
    };
    check(cache.insert(3, 30, nullptr, select_second) == 2, "expected allocation-free alternate eviction");
    check(cache.peek(2) == nullptr && cache.peek(3) && *cache.peek(3) == 30, "recycled entry must use the new key");
    check(cache.insert(4, 40) == 1, "SIEVE hand must resume after the actual victim");
    check(cache.size() == 2, "recycled entries must preserve the item budget");
}

struct ValueWithThrowingAssignment {
    int value;

    explicit ValueWithThrowingAssignment(int value) : value(value) {}
    ValueWithThrowingAssignment(ValueWithThrowingAssignment&&) = default;
    ValueWithThrowingAssignment& operator=(ValueWithThrowingAssignment&& other) noexcept(false) {
        value = other.value;
        return *this;
    }
};

void check_staged_insert_failure() {
    bool blocked = false;
    Cache<ValueWithThrowingAssignment, TestAllocator> cache(
        1, Policy::sieve, TestAllocator<ValueWithThrowingAssignment>(blocked));
    cache.insert(1, ValueWithThrowingAssignment(10));

    blocked = true;
    try {
        cache.insert(2, ValueWithThrowingAssignment(20));
        check(false, "expected staging to allocate");
    } catch (const std::bad_alloc&) {
        check(cache.size() == 1 && cache.peek(1) && cache.peek(1)->value == 10,
              "failed staging must preserve the resident");
        check(cache.peek(2) == nullptr, "failed staging must not publish the new entry");
    }
    blocked = false;
    check(cache.insert(2, ValueWithThrowingAssignment(20)) == 1, "staged insert should succeed on retry");
}

}  // namespace

int main() {
    check_example();
    check_sieve_wrap();
    check_recycled_nodes();
    check_staged_insert_failure();
    for (auto policy : {Policy::lru, Policy::sieve}) {
        check_small_capacities(policy);
        check_update(policy);
    }
}
