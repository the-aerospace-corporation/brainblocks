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
    uint32_t d_thresh;   // coincidence detector threshold
    uint32_t perm_thr;   // permanence increment
    uint32_t perm_inc;   // permanence increment
    uint32_t perm_dec;   // permanence decrement
	uint32_t count_hs;   // historical statelet counter
	uint32_t count_hd;   // historical coincidence detector counter
    double pct_score;    // abnormality score percentage (0.0 to 1.0)
    uint8_t init_flag;   // initialized flag
    uint32_t* s_next_d;  // array of next available coincidence detector index for each statelet
    struct Page* input;  // input page object
    struct Page* hidden; // hidden page object
	struct Page* output; // output page object
    struct BitArray* connections_ba; // connections bitarray object (helper for overlap function)
    struct BitArray* activeconns_ba; // active connections bitarray object (helper for overlap function)
    struct CoincidenceSet* d_hidden; // array of hidden coincidence set objects
	struct CoincidenceSet* d_output; // array of output coincidence set objects
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

struct BitArray* sequence_learner_get_historical_statelets(struct SequenceLearner* sl);

void sequence_learner_overlap(
    struct SequenceLearner* sl,
    const struct ActArray* input_aa);

void sequence_learner_activate(
    struct SequenceLearner* sl,
    const struct ActArray* input_aa,
    const uint32_t learn_flag);

void sequence_learner_learn(
    struct SequenceLearner* sl,
    const struct ActArray* input_aa);

#endif