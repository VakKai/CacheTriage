#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace cachetriage {

using Key = std::uint64_t;

enum class Policy { lru, sieve };

struct EvictionShortlist {
    std::array<Key, 4> keys{};
    std::size_t size = 0;
};

// A single-threaded cache bounded by its number of resident entries.
template <typename Value, template <typename> class Allocator = std::allocator>
class Cache {
public:
    using ShortlistValues = std::array<const Value*, std::tuple_size_v<decltype(EvictionShortlist::keys)>>;

    Cache(std::size_t capacity, Policy policy, Allocator<Value> allocator = {})
        : capacity_(capacity), policy_(policy), entries_(Allocator<Entry>(allocator)),
          index_(Allocator<IndexValue>(allocator)), hand_(entries_.end()) {}

    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;

    // A hit updates policy metadata. The pointer is valid until eviction or destruction.
    Value* find(Key key) {
        const auto found = index_.find(key);
        if (found == index_.end()) {
            return nullptr;
        }
        touch(found->second);
        return std::addressof(found->second->value);
    }

    // Inspect a resident value without updating its eviction metadata.
    const Value* peek(Key key) const {
        const auto found = index_.find(key);
        return found == index_.end() ? nullptr : std::addressof(found->second->value);
    }

    // Replacing a resident value counts as a touch. Returns the victim, if any.
    // Optionally record the shortlist or select a victim by its index in that list.
    // A nonthrowing selector may also accept the corresponding resident value pointers.
    // The pointers are borrowed for the call. An out-of-range index keeps the default victim.
    template <typename Select = std::nullptr_t>
    std::optional<Key> insert(Key key, Value value, EvictionShortlist* shortlist = nullptr,
                              Select select = nullptr) {
        if (capacity_ == 0) {
            return std::nullopt;
        }
        const auto found = index_.find(key);
        if (found != index_.end()) {
            found->second->value = std::move(value);
            touch(found->second);
            return std::nullopt;
        }

        if constexpr (std::is_nothrow_move_assignable_v<Value>) {
            if (entries_.size() == capacity_) {
                const auto victim = pick_victim(shortlist, select);
                const auto evicted = victim->key;
                // Re-key the existing hash node. Restoring the previous size
                // fits the existing bucket array, so there is no rehash.
                auto slot = index_.extract(evicted);
                slot.key() = key;
                index_.insert(std::move(slot));
                victim->key = key;
                victim->value = std::move(value);
                victim->visited = false;
                entries_.splice(entries_.end(), entries_, victim);
                return evicted;
            }
        }

        // Stage the new entry before changing policy state. Splicing preserves
        // its iterator and keeps it out of the eviction scan until admission.
        Entries pending(entries_.get_allocator());
        pending.push_back({key, std::move(value), false});
        index_.emplace(key, pending.begin());

        std::optional<Key> evicted;
        if (entries_.size() == capacity_) {
            const auto victim = pick_victim(shortlist, select);
            evicted = victim->key;
            index_.erase(*evicted);
            entries_.erase(victim);
        }
        entries_.splice(entries_.end(), pending);
        return evicted;
    }

    std::size_t size() const { return entries_.size(); }

    // Offline replay can branch from the exact list order, visited bits, and hand.
    // Requires copyable values. The new index contains only iterators into the fork.
    std::unique_ptr<Cache> fork_for_replay() const {
        auto copy = std::make_unique<Cache>(capacity_, policy_, Allocator<Value>(entries_.get_allocator()));
        copy->index_.reserve(index_.size());
        for (auto entry = entries_.cbegin(); entry != entries_.cend(); ++entry) {
            copy->entries_.push_back(*entry);
            const auto position = std::prev(copy->entries_.end());
            copy->index_.emplace(entry->key, position);
            if (entry == hand_) {
                copy->hand_ = position;
            }
        }
        return copy;
    }

private:
    struct Entry {
        Key key;
        Value value;
        bool visited;
    };
    using Entries = std::list<Entry, Allocator<Entry>>;
    using Position = typename Entries::iterator;
    using IndexValue = std::pair<const Key, Position>;

    void touch(Position entry) {
        if (policy_ == Policy::lru) {
            entries_.splice(entries_.end(), entries_, entry);
        } else {
            entry->visited = true;
        }
    }

    void record_shortlist(Position victim, EvictionShortlist& shortlist, ShortlistValues* values) {
        shortlist.size = 0;
        auto entry = victim;
        do {
            if (values) {
                (*values)[shortlist.size] = std::addressof(entry->value);
            }
            shortlist.keys[shortlist.size++] = entry->key;
            if (++entry == entries_.end()) {
                entry = entries_.begin();
            }
        } while (entry != victim && shortlist.size < shortlist.keys.size());
    }

    template <typename Select>
    Position pick_victim(EvictionShortlist* shortlist, Select& select) {
        constexpr bool has_selector = !std::is_same_v<Select, std::nullptr_t>;
        constexpr bool selects_keys =
            std::is_nothrow_invocable_r_v<std::size_t, Select&, const EvictionShortlist&>;
        constexpr bool selects_values = !selects_keys &&
            std::is_nothrow_invocable_r_v<std::size_t, Select&, const EvictionShortlist&,
                                          const ShortlistValues&>;
        static_assert(!has_selector || selects_keys || selects_values,
                      "Victim selector must return an index without throwing");
        auto victim = choose_victim();
        if (shortlist || has_selector) {
            EvictionShortlist candidates;
            ShortlistValues values;
            record_shortlist(victim, candidates, selects_values ? &values : nullptr);
            if (shortlist) {
                *shortlist = candidates;
            }
            if constexpr (has_selector) {
                const std::size_t selected = [&] {
                    if constexpr (selects_values) {
                        return select(std::as_const(candidates), std::as_const(values));
                    } else {
                        return select(std::as_const(candidates));
                    }
                }();
                if (selected != 0 && selected < candidates.size) {
                    victim = index_.find(candidates.keys[selected])->second;
                }
            }
        }
        if (policy_ == Policy::sieve) {
            // If the victim was newest, restart at the oldest on the next scan.
            hand_ = std::next(victim);
        }
        return victim;
    }

    Position choose_victim() {
        if (policy_ == Policy::lru) {
            return entries_.begin();
        }

        auto candidate = hand_ == entries_.end() ? entries_.begin() : hand_;
        while (candidate->visited) {
            candidate->visited = false;
            if (++candidate == entries_.end()) {
                candidate = entries_.begin();
            }
        }
        return candidate;
    }

    std::size_t capacity_;
    Policy policy_;
    // Oldest at the front, newest at the back. Only LRU reorders on hits.
    Entries entries_;
    std::unordered_map<Key, Position, std::hash<Key>, std::equal_to<Key>, Allocator<IndexValue>> index_;
    Position hand_;
};

}  // namespace cachetriage
