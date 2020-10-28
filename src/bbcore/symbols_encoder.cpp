#include "symbols_encoder.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
SymbolsEncoder::SymbolsEncoder(
        const uint32_t max_symbols,
        const uint32_t num_s) {

    // error check
    if (max_symbols == 0) {
        std::cout << "Error in SymbolsEncoder::SymbolsEncoder: max_symbols == 0" << std::endl;
        exit(1);
    }   
    
    if (num_s == 0) {
        std::cout << "Error in SymbolsEncoder::SymbolsEncoder: num_s == 0" << std::endl;
        exit(1);
    }

    if (max_symbols > num_s) {
        std::cout << "Error in SymbolsEncoder::SymbolsEncoder: max_symbols > num_s" << std::endl;
        exit(1);
    }

    // setup variables
    this->max_symbols = max_symbols;
    this->num_s = num_s;
    this->num_as = (uint32_t)((double)(num_s) / (double)(max_symbols));
    this->range_bits = num_s;
    this->init_flag = false;
    
    symbols.resize(max_symbols);

    // setup symbols array
    for (uint32_t i = 0; i < max_symbols; i++) {
        symbols[i] = i;
    }

    // setup pages
    output.set_num_history(2);
    output.set_num_bits(num_s);
}

// =============================================================================
// Initialize
// =============================================================================
void SymbolsEncoder::initialize() {
    output.initialize();
    init_flag = true;
}

// =============================================================================
// Clear
// =============================================================================
void SymbolsEncoder::clear() {
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}

// =============================================================================
// Compute
// =============================================================================
void SymbolsEncoder::compute(const uint32_t value) {
    if (init_flag == false) {
        initialize();
    }

    output.step();

    bool compute_flag = true;
    bool new_flag = true;

    if (value >= max_symbols) {
        std::cout << "Warning in SymbolsEncoder::compute(): value not in symbols" << std::endl;
        output[CURR].clear_bits();
        return;
    }

    double percent = (double)value / (double)max_symbols;
    uint32_t beg = (uint32_t)((double)range_bits * percent);
    uint32_t end = beg + num_as - 1;
    
    for (uint32_t i = beg; i <= end; i++) {
        output[CURR].set_bit(i, 1);
    }

    output.compute_changed();
}