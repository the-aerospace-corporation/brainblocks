#ifndef PATTERN_POOLER_H
#define PATTERN_POOLER_H

#include "bitarray.h"
#include "page.h"
#include "coincidence_set.h"
#include <stdint.h>

struct PatternPooler {
    uint32_t num_s;       // number of statelets
    uint32_t num_as;      // number of active statelets
    uint32_t perm_thr;    // permanence threshold
    uint32_t perm_inc;    // permanence increment
    uint32_t perm_dec;    // permanence decrement
    double pct_pool;      // percent pool
    double pct_conn;      // percent initially connected
    double pct_learn;     // percent learn
    uint8_t init_flag;    // initialized flag
    uint32_t* learn_mask; // receptor random learning mask array
    struct Page* input;   // input page object
    struct Page* output;  // output page object
    struct CoincidenceSet* coincidence_sets;
};

void pattern_pooler_construct(
    struct PatternPooler* pp,
    const uint32_t num_s,
    const uint32_t num_as,
    const uint32_t perm_thr,
    const uint32_t perm_inc,
    const uint32_t perm_dec,
    const double pct_pool,
    const double pct_conn,
    const double pct_learn);

void pattern_pooler_destruct(struct PatternPooler* pp);
void pattern_pooler_initialize(struct PatternPooler* pp);
void pattern_pooler_save(struct PatternPooler* pp, const char* file);
void pattern_pooler_load(struct PatternPooler* pp, const char* file);
void pattern_pooler_clear(struct PatternPooler* pp);

void pattern_pooler_compute(
    struct PatternPooler* pp,
    const uint32_t learn_flag);

void pattern_pooler_overlap_(struct PatternPooler* pp);
void pattern_pooler_activate_(struct PatternPooler* pp);
void pattern_pooler_learn_(struct PatternPooler* pp);

#endif