#ifndef SYMBOLS_ENCODER_H
#define SYMBOLS_ENCODER_H

#include "page.h"
#include <stdint.h>

struct SymbolsEncoder {
    uint32_t max_symbols; // maximum number of symbols
    uint32_t num_symbols; // number of symbols
    uint32_t num_s;       // number of statelets
    uint32_t num_as;      // number of active statelets
    uint32_t range_bits;  // bit range
    uint8_t init_flag;    // initialized flag
    uint32_t* symbols;    // symbols
    struct Page* output;  // output page object
};

void symbols_encoder_construct(
    struct SymbolsEncoder* e,
    const uint32_t max_symbols,
    const uint32_t num_s);

void symbols_encoder_destruct(struct SymbolsEncoder* e);
void symbols_encoder_initialize(struct SymbolsEncoder* e);
void symbols_encoder_compute(struct SymbolsEncoder* e, const uint32_t value);

#endif