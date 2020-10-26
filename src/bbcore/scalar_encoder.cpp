#include "scalar_encoder.hpp"
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

// =============================================================================
// Constructor
// =============================================================================
void scalar_encoder_construct(
    struct ScalarEncoder* e,
    const double min_val,
    const double max_val,
    const uint32_t num_s,
    const uint32_t num_as) {

    // error check
    if (min_val > max_val) {
        perror("Error: ScalarEncoder min_val > max_val");
        exit(1);
    }   

    if (num_s == 0) {
        perror("Error: ScalarEncoder num_s == 0");
        exit(1);
    }

    if (num_as == 0) {
        perror("Error: ScalarEncoder num_as == 0");
        exit(1);
    }

    if (num_as > num_s) {
        perror("Error: ScalarEncoder num_as > num_s");
        exit(1);
    }

    // initialize variables
    e->min_val = min_val;
    e->max_val = max_val;
    e->range_val = e->max_val - e->min_val;
    e->num_s = num_s;
    e->num_as = num_as;
    e->range_bits = e->num_s - e->num_as;
    e->init_flag = 0;
    e->output = (Page*)malloc(sizeof(*e->output));

    // construct pages
    page_construct(e->output, 2, e->num_s);
}

// =============================================================================
// Destruct
// =============================================================================
void scalar_encoder_destruct(struct ScalarEncoder* e) {
    page_destruct(e->output);
    free(e->output);
}

// =============================================================================
// Initialize
// =============================================================================
void scalar_encoder_initialize(struct ScalarEncoder* e) {
    page_initialize(e->output);
    e->init_flag = 1;
}

// =============================================================================
// Clear
// =============================================================================
void scalar_encoder_clear(struct ScalarEncoder* e) {
    page_clear_bits(e->output, 0); // current
    page_clear_bits(e->output, 1); // previous
}

// =============================================================================
// Compute
// =============================================================================
void scalar_encoder_compute(struct ScalarEncoder* e, double value) {
    if (e->init_flag == 0) {
        scalar_encoder_initialize(e);
    }

    page_step(e->output);

    if (value < e->min_val) value = e->min_val;
    if (value > e->max_val) value = e->max_val;
    double percent = (value - e->min_val) / e->range_val;
    uint32_t beg = (uint32_t)((double)e->range_bits * percent);
    uint32_t end = beg + e->num_as - 1;

    for (uint32_t i = beg; i <= end; i++) {
        page_set_bit(e->output, 0, i);
    }
    
    page_compute_changed(e->output);
}