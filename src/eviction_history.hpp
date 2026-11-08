#pragma once

#include <cstddef>

namespace cachetriage {

struct EvictionHistory {
    std::size_t admitted;
    std::size_t last;
    std::size_t previous = 0;
    std::size_t hits = 0;

    void record_hit(std::size_t request) {
        previous = last;
        last = request;
        ++hits;
    }
};

}  // namespace cachetriage
