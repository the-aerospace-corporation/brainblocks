#ifndef SEQUENCE_LEARNER_H
#define SEQUENCE_LEARNER_H

#include "bitarray.h"
#include "page.h"
#include "coincidence_set.h"
#include <stdint.h>

struct SequenceLearner {
    uint32_t num_c;      // number of columns
    uint32_t num_spc;    // number of statelets per column
    uint32_t num_dps;    // number of coincidence detectors per statelet
    uint32_t num_dpc;    // number of coincidence detectors per column
    uint32_t num_s;      // number of statelets
    uint32_t num_d;      // number of coincidence detectors
    uint32_t num_rpd;    // number of receptors per coincidence detector
    uint8_t d_thresh;    // coincidence detector threshold
    uint32_t perm_thr;   // permanence increment
    uint32_t perm_inc;   // permanence increment
    uint32_t perm_dec;   // permanence decrement
    double pct_score;    // abnormality pct_score (0.0 to 1.0)
    uint8_t init_flag;   // initialized flag
    uint32_t* n_next_d;  // next available dendrite on each neuron
    struct Page* input;  // input page object
    struct Page* output; // output page object
    struct BitArray* connections_ba;
    struct BitArray* activeconns_ba;
    struct CoincidenceSet* coincidence_sets;
};

void sequence_learner_construct(
    struct SequenceLearner* sl,
    const uint32_t num_spc,
    const uint32_t num_dps,
    const uint32_t num_rpd,
    const uint32_t d_thresh,
    const uint32_t perm_thr,
    const uint32_t perm_inc,
    const uint32_t perm_dec);

void sequence_learner_destruct(struct SequenceLearner* sl);
void sequence_learner_initialize(struct SequenceLearner* sl);
void sequence_learner_save(struct SequenceLearner* sl, const char* file);
void sequence_learner_load(struct SequenceLearner* sl, const char* file);
void sequence_learner_clear(struct SequenceLearner* sl);

void sequence_learner_compute(
    struct SequenceLearner* sl,
    const uint32_t learn_flag);

double sequence_learner_get_score(struct SequenceLearner* sl);

void sequence_learner_overlap_(
    struct SequenceLearner* sl,
    const struct ActArray* input_aa);

void sequence_learner_activate_(
    struct SequenceLearner* sl,
    const struct ActArray* input_aa,
    const uint32_t learn_flag);

void sequence_learner_learn_(
    struct SequenceLearner* sl,
    const struct ActArray* input_aa);

#endif