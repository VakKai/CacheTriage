#pragma once

#include "cachetriage/cache.hpp"

#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace cachetriage {

template <typename UInt>
bool parse_unsigned(std::string_view input, UInt& result) {
    if (input.empty()) {
        return false;
    }
    const auto end = input.data() + input.size();
    const auto parsed = std::from_chars(input.data(), end, result);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

template <typename Read>
void for_each_request(const char* path, Read read) {
    std::ifstream trace(path);
    if (!trace) {
        throw std::runtime_error(std::string("Cannot open trace: ") + path);
    }
    constexpr std::string_view whitespace = " \t\r\n\v\f";
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(trace, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(whitespace);
        if (first == std::string::npos) {
            continue;
        }
        const auto last = line.find_last_not_of(whitespace);
        Key key = 0;
        if (!parse_unsigned(std::string_view(line).substr(first, last - first + 1), key)) {
            throw std::runtime_error(std::string(path) + ':' + std::to_string(line_number) +
                                     ": expected one unsigned 64-bit decimal key");
        }
        read(key);
    }
    if (trace.bad() || !trace.eof()) {
        throw std::runtime_error(std::string("Error reading trace: ") + path);
    }
}

}  // namespace cachetriage
