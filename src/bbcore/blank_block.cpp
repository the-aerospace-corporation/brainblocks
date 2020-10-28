#include "blank_block.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
BlankBlock::BlankBlock(const uint32_t num_s) {
    id = 0;
    seed = 0;
    this->num_s = num_s;
    output.set_num_history(2);
    output.set_num_bits(num_s);
    output.initialize();
}

// =============================================================================
// Clear
// =============================================================================
void BlankBlock::clear() {
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}