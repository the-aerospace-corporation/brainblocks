#ifndef SYMBOLS_ENCODER_HPP
#define SYMBOLS_ENCODER_HPP

#include "page.hpp"
#include <cstdint>
#include <vector>

class SymbolsEncoder {
    public:
        SymbolsEncoder(
            const uint32_t max_symbols,
            const uint32_t num_s);

        void initialize();
        void clear();
        void compute(uint32_t value);

    public:
        Page output; // output page object

    private:
        uint32_t max_symbols; // maximum number of symbols
        uint32_t num_s;       // number of statelets
        uint32_t num_as;      // number of active statelets
        uint32_t range_bits;  // bit range
        bool init_flag;       // initialized flag
        std::vector<uint32_t> symbols;
};

#endif