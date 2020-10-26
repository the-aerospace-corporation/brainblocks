#ifndef SCALAR_ENCODER_HPP
#define SCALAR_ENCODER_HPP

#include "page.hpp"
#include <stdint.h>

struct ScalarEncoder {
    double min_val;      // maximum input value
    double max_val;      // minimum input value
    double range_val;    // value range
    uint32_t num_s;      // number of statelets
    uint32_t num_as;     // number of active statelets
    uint32_t range_bits; // bit range
    uint8_t init_flag;   // initialized flag
    struct Page* output; // output page object
};

void scalar_encoder_construct(
    struct ScalarEncoder* e,
    const double min_val,
    const double max_val,
    const uint32_t num_s,
    const uint32_t num_as);

void scalar_encoder_destruct(struct ScalarEncoder* e);
void scalar_encoder_initialize(struct ScalarEncoder* e);
void scalar_encoder_clear(struct ScalarEncoder* e);
void scalar_encoder_compute(struct ScalarEncoder* e, double value);

#endif