#include "blank_block.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
BlankBlock::BlankBlock(const uint32_t num_s) {
    id = 0;
    seed = 0;
    this->num_s = num_s;
    output = new Page(2, num_s);
    output->initialize();
}

// =============================================================================
// Destructor
// =============================================================================
BlankBlock::~BlankBlock() {
    delete output;
}

// =============================================================================
// Print Parameters
// =============================================================================
void BlankBlock::print_parameters() {
    std::cout << "Block Parameters:" << std::endl;
    std::cout << "- id: " << id << std::endl;
    std::cout << "- type: BlankBlock" << std::endl;
    std::cout << "- num_s: " << num_s << std::endl;
    std::cout << std::endl;
}

// =============================================================================
// Clear
// =============================================================================
void BlankBlock::clear() {
    output->clear_bits(0); // current
    output->clear_bits(1); // previous
}