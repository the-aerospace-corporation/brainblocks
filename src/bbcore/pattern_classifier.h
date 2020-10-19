#ifndef PATTERN_CLASSIFIER_H
#define PATTERN_CLASSIFIER_H

#include "bitarray.h"
#include "page.h"
#include "coincidence_set.h"
#include <stdint.h>

struct PatternClassifier {
    uint32_t num_l;       // number of labels
    uint32_t num_s;       // number of statelets
    uint32_t num_as;      // number of active statelets
    uint32_t num_spl;     // number of states per label
    uint32_t perm_thr;    // permanence threshold
    uint32_t perm_inc;    // permanence increment
    uint32_t perm_dec;    // permanence decrement
    double pct_pool;      // percent pool
    double pct_conn;      // percent initially connected
    double pct_learn;     // percent learn
    uint8_t init_flag;    // initialized flag
    uint32_t* s_labels;   // statelet labels
    uint32_t* learn_mask; // receptor random learning mask array
    uint32_t* labels;     // labels
    double* probs;        // probabilities array
    struct Page* input;   // input page object
    struct Page* output;  // output page object
    struct CoincidenceSet* coincidence_sets;
};

void pattern_classifier_construct(
    struct PatternClassifier* pc,
    const uint32_t* labels,
    const uint32_t num_l,
    const uint32_t num_s,
    const uint32_t num_as,
    const uint32_t perm_thr,
    const uint32_t perm_inc,
    const uint32_t perm_dec,
    const double pct_pool,
    const double pct_conn,
    const double pct_learn);

void pattern_classifier_destruct(struct PatternClassifier* pc);
void pattern_classifier_initialize(struct PatternClassifier* pc);
void pattern_classifier_save(struct PatternClassifier* pc, const char* file);
void pattern_classifier_load(struct PatternClassifier* pc, const char* file);
void pattern_classifier_clear(struct PatternClassifier* pc);

void pattern_classifier_compute(
        struct PatternClassifier* pc,
        const uint32_t in_label,
        const uint32_t learn_flag);

void pattern_classifier_update_probabilities(struct PatternClassifier* pc);

void pattern_classifier_overlap_(struct PatternClassifier* pc);
void pattern_classifier_activate_(struct PatternClassifier* pc);

void pattern_classifier_learn_(
    struct PatternClassifier* pc,
    const uint32_t in_label);

struct BitArray* pattern_classifier_decode(struct PatternClassifier* pc);

#endif