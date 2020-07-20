#include "persistence_encoder.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
void persistence_encoder_construct(
    struct PersistenceEncoder* e,
    const double min_val,
    const double max_val,
    const uint32_t num_s,
    const uint32_t num_as,
    const uint32_t max_steps) {

    // error check
    if (min_val > max_val) {
        perror("Error: PersistenceEncoder min_val > max_val");
        exit(1);
    }   

    if (num_s == 0) {
        perror("Error: PersistenceEncoder num_s == 0");
        exit(1);
    }

    if (num_as == 0) {
        perror("Error: PersistenceEncoder num_as == 0");
        exit(1);
    }

    if (num_as > num_s) {
        perror("Error: PersistenceEncoder num_as > num_s");
        exit(1);
    }

    if (max_steps == 0) {
        perror("Error: PersistenceEncoder max_steps == 0");
        exit(1);
    }

    // initialize variables
    e->min_val = min_val;
    e->max_val = max_val;
    e->range_val = e->max_val - e->min_val;
    e->num_s = num_s;
    e->num_as = num_as;
    e->range_bits = e->num_s - e->num_as;
    e->max_steps = max_steps;
    e->step = 0;
    e->pct_val_prev = 0.0;
    e->init_flag = 0;
    e->output = malloc(sizeof(*e->output));

    // construct pages
    page_construct(e->output, 2, e->num_s);
}

// =============================================================================
// Destruct
// =============================================================================
void persistence_encoder_destruct(struct PersistenceEncoder* e) {
    page_destruct(e->output);
    free(e->output);
}

// =============================================================================
// Initialize
// =============================================================================
void persistence_encoder_initialize(struct PersistenceEncoder* e) {
    page_initialize(e->output);
    e->init_flag = 1;
}

// =============================================================================
// Compute
// =============================================================================
void persistence_encoder_compute(struct PersistenceEncoder* e, double value) {
    if (e->init_flag == 0) {
        persistence_encoder_initialize(e);
    }

    page_step(e->output);

    if (value < e->min_val) value = e->min_val;
    if (value > e->max_val) value = e->max_val;

    double percent_time = (double)e->step / (double)e->max_steps;
    double pct_value = (value - e->min_val) / e->range_val;
    double pct_delta = pct_value - e->pct_val_prev;
    uint32_t reset_timer_flag = 0;

    // TODO: may not work if value change is too small over a long period of time
    if (fabs(pct_delta) <= 0.1) { // TODO: percent_reset = 0.1
        e->step += 1;
    }
    else {
        reset_timer_flag = 1;
    }
    
    if (e->step >= e->max_steps) {
        e->step = e->max_steps;
    }
    
    if (reset_timer_flag) {
        e->step = 0;
        e->pct_val_prev = pct_value;
    }

    uint32_t beg = (uint32_t)((double)e->range_bits * percent_time);
    uint32_t end = beg + e->num_as - 1;
    
    for (uint32_t i = beg; i <= end; i++) {
        page_set_bit(e->output, 0, i);
    }
    
    page_compute_changed(e->output);
}

// =============================================================================
// Reset
// =============================================================================
void persistence_encoder_reset(struct PersistenceEncoder* e) {
    e->step = 0;
    e->pct_val_prev = 0.0;
}