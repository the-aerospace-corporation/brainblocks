#ifndef BLANK_BLOCK_HPP
#define BLANK_BLOCK_HPP

#include "page.hpp"
#include <cstdint>

class BlankBlock {
    public:
        BlankBlock(const uint32_t num_s);
        void clear();

    public:
        Page output;

    private:
        uint32_t id;
        uint32_t seed;
        uint32_t num_s; // number of statelets
};

#endif