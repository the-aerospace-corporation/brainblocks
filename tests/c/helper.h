#ifndef HELPER_H
#define HELPER_H

#include "bitarray.h"
#include "page.h"
#include "coincidence_set.h"
#include "blank_block.h"
#include "scalar_encoder.h"
#include "symbols_encoder.h"
#include "persistence_encoder.h"
#include "pattern_classifier.h"
#include "pattern_pooler.h"
#include "sequence_learner.h"

#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


void bitarray_print_bits(struct BitArray* ba) {
    printf("{");
    for (uint32_t i = ba->num_bits; i-- > 0; ) {
        if (bitarray_get_bit(ba, i)) {
            printf("1");
        }
        else {
            printf("0");
        }
    }
    printf("}\n");
}


void bitarray_print_acts(struct BitArray* ba) {
    struct ActArray* aa = bitarray_get_actarray(ba);
    printf("{");
    for (uint32_t i = 0; i < aa->num_acts; i++) {
        if (i > 0) {
            printf(" ");
        }
        printf("%i", aa->acts[i]);
    }
    printf("}\n");
}


void page_print_bits(struct Page* p, const uint32_t t) {
    uint32_t i = page_idx_(p, t);
    bitarray_print_bits(p->bitarrays[i]);
}


void page_print_acts(struct Page* p, const uint32_t t) {
    uint32_t i = page_idx_(p, t);
    bitarray_print_acts(p->bitarrays[i]);
}


void coincidence_set_print(struct CoincidenceSet* d) {
    printf("{");
    for (uint32_t s = 0; s < d->num_r; s++) {
        if (s > 0) {
            printf(" ");
        }
        printf("%i(%i)", d->addrs[s], d->perms[s]);
    }
    printf("}\n");
}


void blank_block_print_parameters(struct BlankBlock* b) {
    printf("BlankBlock\n");
    printf("  num_s = %i\n", b->num_s);
}


void scalar_encoder_print_parameters(struct ScalarEncoder* e) {
    printf("ScalarEncoder\n");
    printf("  min_val = %f\n", e->min_val);
    printf("  max_val = %f\n", e->max_val);
    printf("  num_s = %i\n", e->num_s);
    printf("  num_as = %i\n", e->num_as);
}


void symbols_encoder_print_parameters(struct SymbolsEncoder* e) {
    printf("SymbolsEncoder\n");
    printf("  max_symbols= %i\n", e->max_symbols);
    printf("  num_s = %i\n", e->num_s);
    printf("  num_as = %i\n", e->num_as);
}


void persistence_encoder_print_parameters(struct PersistenceEncoder* e) {
    printf("PersistenceEncoder\n");
    printf("  min_val = %f\n", e->min_val);
    printf("  max_val = %f\n", e->max_val);
    printf("  num_s = %i\n", e->num_s);
    printf("  num_as = %i\n", e->num_as);
    printf("  max_steps = %i\n", e->max_steps);
}


void pattern_classifier_print_parameters(struct PatternClassifier* pc) {
    printf("PatternClassifier\n");
    printf("  labels = {");
    for (uint32_t l = 0; l < pc->num_l; l++) {
        if (l > 0) {
            printf(", ");
        }
        printf("%i", pc->labels[l]);
    }
    printf("}\n");
    printf("  num_l = %i\n", pc->num_l);
    printf("  num_s = %i\n", pc->num_s);
    printf("  num_as = %i\n", pc->num_as);
    printf("  num_spl = %i\n", pc->num_spl);
    printf("  perm_thr = %i\n", pc->perm_thr);
    printf("  perm_inc = %i\n", pc->perm_inc);
    printf("  perm_dec = %i\n", pc->perm_dec);
    printf("  pct_pool = %f\n", pc->pct_pool);
    printf("  pct_conn = %f\n", pc->pct_conn);
    printf("  pct_learn = %f\n", pc->pct_learn);
}


void pattern_classifier_print_probabilities(struct PatternClassifier* pc) {
    pattern_classifier_update_probabilities(pc);

    printf("{");
    for (uint32_t l = 0; l < pc->num_l; l++) {
        if (l > 0) {
            printf(" ");
        }
        printf("%i=%f", pc->labels[l], pc->probs[l]);
    }
    printf("}\n");
}


void pattern_classifier_print_neuron(const struct PatternClassifier* pc, const uint32_t d) {
    bitarray_print_bits(pc->coincidence_sets[d].connections_ba);
}


void pattern_classifier_print_state_overlaps(struct PatternClassifier* pc) {
    printf("{");
    for (uint32_t s = 0; s < pc->num_s; s++) {
        if (s > 0) {
            printf(" ");
        }
        printf("%i", pc->coincidence_sets[s].overlap);
    }
    printf("}\n");
}


void pattern_classifier_print_state_labels(struct PatternClassifier* pc) {
    printf("{");
    for (uint32_t s = pc->num_s; s-- > 0; ) {
        printf("%i", pc->labels[pc->s_labels[s]]);
    }
    printf("}\n");
}


void pattern_pooler_print_parameters(struct PatternPooler* pp) {
    printf("PatternPooler\n");
    printf("  num_s = %i\n", pp->num_s);
    printf("  num_as = %i\n", pp->num_as);
    printf("  perm_thr = %i\n", pp->perm_thr);
    printf("  perm_inc = %i\n", pp->perm_inc);
    printf("  perm_dec = %i\n", pp->perm_dec);
    printf("  pct_pool = %f\n", pp->pct_pool);
    printf("  pct_conn = %f\n", pp->pct_conn);
    printf("  pct_learn = %f\n", pp->pct_learn);
}


void pattern_pooler_print_overlaps(struct PatternPooler* pp) {
    printf("{");
    for (uint32_t s = 0; s < pp->num_s; s++) {
        if (s > 0) {
            printf(" ");
        }
        printf("%i", pp->coincidence_sets[s].overlap);
    }
    printf("}\n");
}


void sequence_learner_print_parameters(struct SequenceLearner* sl) {
    printf("SequenceLearner\n");
    printf("  num_c = %i\n", sl->num_c);
    printf("  num_spc = %i\n", sl->num_spc);
    printf("  num_dps = %i\n", sl->num_dps);
    printf("  num_dpc = %i\n", sl->num_dpc);
    printf("  num_s = %i\n", sl->num_s);
    printf("  num_d = %i\n", sl->num_d);
    printf("  num_rpd = %i\n", sl->num_rpd);
    printf("  d_thresh = %i\n", sl->d_thresh);
    printf("  perm_thr = %i\n", sl->perm_thr);
    printf("  perm_inc = %i\n", sl->perm_inc);
    printf("  perm_dec = %i\n", sl->perm_dec);
}

#endif