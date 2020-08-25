#ifndef PERSISTENCE_ENCODER_H
#define PERSISTENCE_ENCODER_H

#include "page.h"
#include <stdint.h>

struct PersistenceEncoder {
    double min_val;      // minimum input value
    double max_val;      // maximum input value
    double range_val;    // value range
    uint32_t num_s;      // number of statelets
    uint32_t num_as;     // number of active statelets
    uint32_t range_bits; // bit range
    uint32_t max_steps;  // maximum steps
    uint32_t step;       // step counter
    double pct_val_prev; // value previous percentage
    uint8_t init_flag;   // initialized flag
    struct Page* output; // output page object
};

void persistence_encoder_construct(
    struct PersistenceEncoder* e,
    const double min_val,
    const double max_val,
    const uint32_t num_s,
    const uint32_t num_as,
    const uint32_t max_steps);

void persistence_encoder_destruct(struct PersistenceEncoder* e);
void persistence_encoder_initialize(struct PersistenceEncoder* e);
void persistence_encoder_clear(struct PersistenceEncoder* e);
void persistence_encoder_compute(struct PersistenceEncoder* e, double value);
void persistence_encoder_reset(struct PersistenceEncoder* e);

#endif