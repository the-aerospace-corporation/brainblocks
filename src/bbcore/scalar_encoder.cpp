#include "scalar_encoder.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
ScalarEncoder::ScalarEncoder(
        const double min_val,
        const double max_val,
        const uint32_t num_s,
        const uint32_t num_as) {

    // error check
    if (min_val > max_val) {
        std::cout << "Error in ScalarEncoder::ScalarEncoder: min_val > max_val" << std::endl;
        exit(1);
    }   

    if (num_s == 0) {
        std::cout << "Error in ScalarEncoder::ScalarEncoder: num_s == 0" << std::endl;
        exit(1);
    }

    if (num_as == 0) {
        std::cout << "Error in ScalarEncoder::ScalarEncoder: num_as == 0" << std::endl;
        exit(1);
    }

    if (num_as > num_s) {
        std::cout << "Error in ScalarEncoder::ScalarEncoder: num_as > num_s" << std::endl;
        exit(1);
    }

    // setup variables
    this->min_val = min_val;
    this->max_val = max_val;
    this->range_val = max_val - min_val;
    this->num_s = num_s;
    this->num_as = num_as;
    this->range_bits = num_s - num_as;
    this->init_flag = false;

    // setup pages
    output.set_num_history(2);
    output.set_num_bits(num_s);
}

// =============================================================================
// Initialize
// =============================================================================
void ScalarEncoder::initialize() {
    output.initialize();
    init_flag = true;
}

// =============================================================================
// Clear
// =============================================================================
void ScalarEncoder::clear() {
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}

// =============================================================================
// Compute
// =============================================================================
void ScalarEncoder::compute(double value) {
    if (init_flag == false) {
        initialize();
    }

    output.step();

    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
    double percent = (value - min_val) / range_val;
    uint32_t beg = (uint32_t)((double)range_bits * percent);
    uint32_t end = beg + num_as - 1;

    for (uint32_t i = beg; i <= end; i++) {
        output[CURR].set_bit(i, 1);
    }
    
    output.compute_changed();
}