#include "symbols_encoder.h"

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
void symbols_encoder_construct(
        struct SymbolsEncoder* e,
        const uint32_t max_symbols,
        const uint32_t num_s) {

    // error check
    if (max_symbols == 0) {
        perror("Error: SymbolsEncoder max_symbols == 0");
        exit(1);
    }   
    
    if (num_s == 0) {
        perror("Error: SymbolsEncoder num_s == 0");
        exit(1);
    }

    if (max_symbols > num_s) {
        perror("Error: SymbolsEncoder max_symbols > num_s");
        exit(1);
    }

    // initialize variables
    e->max_symbols = max_symbols;
    e->num_symbols = 0;
    e->num_s = num_s;
    e->num_as = (uint32_t)((double)(e->num_s) / (double)(max_symbols));
    e->range_bits = e->num_s;
    e->init_flag = 0;
    e->symbols = malloc(e->max_symbols * sizeof(*e->symbols));    
    e->output = malloc(sizeof(*e->output));

    // construct pages
    page_construct(e->output, 2, e->num_s);

    // initialize symbols array
    for (uint32_t i = 0; i < e->max_symbols; i++) {
        e->symbols[i] = i;
    }
}

// =============================================================================
// Destruct
// =============================================================================
void symbols_encoder_destruct(struct SymbolsEncoder* e) {
    page_destruct(e->output);

    free(e->output);
    free(e->symbols);
}

// =============================================================================
// Initialize
// =============================================================================
void symbols_encoder_initialize(struct SymbolsEncoder* e) {
    page_initialize(e->output);
    e->init_flag = 1;
}

// =============================================================================
// Clear
// =============================================================================
void symbols_encoder_clear(struct SymbolsEncoder* e) {
    page_clear_bits(e->output, 0); // current
    page_clear_bits(e->output, 1); // previous
}

// =============================================================================
// Compute
// =============================================================================
void symbols_encoder_compute(struct SymbolsEncoder* e, const uint32_t value) {
    if (e->init_flag == 0) {
        symbols_encoder_initialize(e);
    }

    page_step(e->output);

    uint32_t compute_flag = 1;
    uint32_t new_flag = 1;

    for (uint32_t i = 0; i < e->num_symbols; i++) {
        if (e->symbols[i] == value) {
            new_flag = 0;
            break;
        }
    }

    if (new_flag) {
        if (e->num_symbols == e->max_symbols) {
            perror("Error: symbols encoder has reached maximum recognized symbols allocation\n");
            compute_flag = 0;
        }
        else {
            e->symbols[e->num_symbols] = value;
            e->num_symbols++;
        }
    }

    if (compute_flag) {
        double percent = (double)value / (double)e->max_symbols;
        uint32_t beg = (uint32_t)((double)e->range_bits * percent);
        uint32_t end = beg + e->num_as - 1;
        
        for (uint32_t i = beg; i <= end; i++) {
            page_set_bit(e->output, 0, i);
        }
    }

    page_compute_changed(e->output);
}