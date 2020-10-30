#include "persistence_encoder.hpp"
#include <iostream>
#include <cmath>

// =============================================================================
// Constructor
// =============================================================================
PersistenceEncoder::PersistenceEncoder(
    const double min_val,
    const double max_val,
    const uint32_t num_s,
    const uint32_t num_as,
    const uint32_t max_steps) {

    // error check
    if (min_val > max_val) {
        std::cout << "Error in PersistenceEncoder::PersistenceEncoder: min_val > max_val" << std::endl;
        exit(1);
    }   

    if (num_s == 0) {
        std::cout << "Error in PersistenceEncoder::PersistenceEncoder: num_s == 0" << std::endl;
        exit(1);
    }

    if (num_as == 0) {
        std::cout << "Error in PersistenceEncoder::PersistenceEncoder: num_as == 0" << std::endl;
        exit(1);
    }

    if (num_as > num_s) {
        std::cout << "Error in PersistenceEncoder::PersistenceEncoder: num_as > num_s" << std::endl;
        exit(1);
    }

    if (max_steps == 0) {
        std::cout << "Error in PersistenceEncoder::PersistenceEncoder: max_steps == 0" << std::endl;
        exit(1);
    }

    // setup variables
    this->min_val = min_val;
    this->max_val = max_val;
    this->range_val = max_val - min_val;
    this->num_s = num_s;
    this->num_as = num_as;
    this->range_bits = num_s - num_as;
    this->max_steps = max_steps;
    this->step = 0;
    this->pct_val_prev = 0.0;
    this->init_flag = false;

    // setup pages
    output.set_num_bitarrays(2);
    output.set_num_bits(num_s);
}

// =============================================================================
// Initialize
// =============================================================================
void PersistenceEncoder::initialize() {
    output.initialize();
    init_flag = true;
}

// =============================================================================
// Clear States
// =============================================================================
void PersistenceEncoder::clear_states() {
    output[CURR].clear_bits();
    output[PREV].clear_bits();

    step = 0;
    pct_val_prev = 0.0;
}

// =============================================================================
// Compute
// =============================================================================
void PersistenceEncoder::compute(double value) {
    if (init_flag == false) {
        initialize();
    }

    output.step();

    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;

    double percent_time = (double)step / (double)max_steps;
    double pct_value = (value - min_val) / range_val;
    double pct_delta = pct_value - pct_val_prev;
    uint32_t reset_timer_flag = 0;

    // TODO: may not work if value change is too small over a long period of time
    if (fabs(pct_delta) <= 0.1) { // TODO: percent_reset = 0.1
        step += 1;
    }
    else {
        reset_timer_flag = 1;
    }
    
    if (step >= max_steps) {
        step = max_steps;
    }
    
    if (reset_timer_flag) {
        step = 0;
        pct_val_prev = pct_value;
    }

    uint32_t beg = (uint32_t)((double)range_bits * percent_time);
    uint32_t end = beg + num_as - 1;
    
    for (uint32_t i = beg; i <= end; i++) {
        output[CURR].set_bit(i, 1);
    }
    
    output.compute_changed();
}