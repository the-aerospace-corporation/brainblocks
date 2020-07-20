#include "sequence_learner.h"

#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
void sequence_learner_construct(
    struct SequenceLearner* sl,
    const uint32_t num_spc,
    const uint32_t num_dps,
    const uint32_t num_rpd,
    const uint32_t d_thresh,
    const uint32_t perm_thr,
    const uint32_t perm_inc,
    const uint32_t perm_dec) {

    // error check
    if (num_spc == 0) {
        perror("Error: SequenceLearner num_spc == 0");
        exit(1);
    }

    if (num_dps == 0) {
        perror("Error: SequenceLearner num_dps == 0");
        exit(1);
    }

    if (num_rpd == 0) {
        perror("Error: SequenceLearner num_rpd == 0");
        exit(1);
    }

    if (d_thresh > num_rpd) {
        perror("Error: SequenceLearner d_thresh > num_rpd");
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        perror("Error: SequenceLearner perm_thr > 99");
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        perror("Error: SequenceLearner perm_inc > 99");
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        perror("Error: SequenceLearner perm_dec > 99");
        exit(1);
    }


    // initialize variables
    sl->num_c = 0;
    sl->num_spc = num_spc;
    sl->num_dps = num_dps;
    sl->num_dpc = sl->num_spc * sl->num_dps;
    sl->num_s = 0;
    sl->num_d = 0;
    sl->num_rpd = num_rpd;
    sl->d_thresh = d_thresh;
    sl->perm_thr = perm_thr;
    sl->perm_inc = perm_inc;
    sl->perm_dec = perm_dec;
    sl->pct_score = 0.0;
    sl->init_flag = 0;
    sl->n_next_d = NULL;
    sl->input = malloc(sizeof(*sl->input));
    sl->output = malloc(sizeof(*sl->output));
    sl->connections_ba = malloc(sizeof(*sl->connections_ba));
    sl->activeconns_ba = malloc(sizeof(*sl->activeconns_ba));
    sl->coincidence_sets = NULL;

    // construct input page (output constructed in initialize())
    page_construct(sl->input, 2, 0);
}

// =============================================================================
// Destructor
// =============================================================================
void sequence_learner_destruct(struct SequenceLearner* sl) {

    // cleanup initialized pointers if applicable
    if (sl->init_flag == 1) {

        // destruct output page, connections bitarray, and activeconns bitarray
        page_destruct(sl->output);    
        bitarray_destruct(sl->connections_ba);
        bitarray_destruct(sl->activeconns_ba);

        // destruct each element in coincidence_sets
        for (uint32_t d = 0; d < sl->num_d; d++) {
            coincidence_set_destruct(&sl->coincidence_sets[d]);
        }
    }

    // destruct input page
    page_destruct(sl->input);

    // free pointers
    free(sl->n_next_d);
    free(sl->input);
    free(sl->output);
    free(sl->coincidence_sets);
    free(sl->connections_ba);
    free(sl->activeconns_ba);
}

// =============================================================================
// Initialize
// =============================================================================
void sequence_learner_initialize(struct SequenceLearner* sl) {

    // initialize input page
    page_initialize(sl->input);

    // update variables
    sl->num_c = page_get_bitarray(sl->input, 0)->num_bits;
    sl->num_s = sl->num_c * sl->num_spc;
    sl->num_d = sl->num_s * sl->num_dps;

    // construct and initialize output page based on input page num_bits
    page_construct(sl->output, 2, sl->num_s);
    page_initialize(sl->output);

    // construct bitarrays
    bitarray_construct(sl->connections_ba, sl->num_s);
    bitarray_construct(sl->activeconns_ba, sl->num_s);

    // initialize neuron's next dendrite array
    sl->n_next_d = calloc(sl->num_s, sizeof(*sl->n_next_d));

    // construct coincidence_sets
    sl->coincidence_sets = malloc(sl->num_d * sizeof(*sl->coincidence_sets));
    for (uint32_t d = 0; d < sl->num_d; d++) {
        coincidence_set_construct(&sl->coincidence_sets[d], sl->num_rpd);
    }

    // set init_flag to true
    sl->init_flag = 1;
}

// =============================================================================
// Save
// =============================================================================
void sequence_learner_save(struct SequenceLearner* sl, const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"wb")) == NULL) {
       printf("Error: sequence_learner_save() cannot open file");
       exit(1);
    }

    for (uint32_t d = 0; d < sl->num_d; d++) {
        struct CoincidenceSet* cs = &sl->coincidence_sets[d];
        fwrite(cs->addrs, cs->num_r * sizeof(uint32_t), 1, fptr);
        fwrite(cs->perms, cs->num_r * sizeof(int32_t), 1, fptr);
    }

    fwrite(sl->n_next_d, sl->num_s * sizeof(uint32_t), 1, fptr);
    fclose(fptr); 
}

// =============================================================================
// Load
// =============================================================================
void sequence_learner_load(struct SequenceLearner* sl, const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"rb")) == NULL) {
       printf("Error: sequence_learner_load() cannot open file\n");
       exit(1);
    }

    for (uint32_t d = 0; d < sl->num_d; d++) {
        struct CoincidenceSet* cs = &sl->coincidence_sets[d];

        if (fread(cs->addrs, cs->num_r * sizeof(uint32_t), 1, fptr) == 0) {
            printf("Error:\n"); // TODO
        }

        if (fread(cs->perms, cs->num_r * sizeof(int32_t), 1, fptr) == 0) {
            printf("Error:\n"); // TODO
        }
    }

    if (fread(sl->n_next_d, sl->num_s * sizeof(uint32_t), 1, fptr) == 0) {
        printf("Error:\n"); // TODO
    }

    fclose(fptr); 
}

// =============================================================================
// Clear
// =============================================================================
void sequence_learner_clear(struct SequenceLearner* sl) {
    page_clear_bits(sl->input, 0); // current
    page_clear_bits(sl->input, 1); // previous
    page_clear_bits(sl->output, 0); // current
    page_clear_bits(sl->output, 1); // previous
}

// =============================================================================
// Compute
// =============================================================================
void sequence_learner_compute(
        struct SequenceLearner* sl,
        const uint32_t learn_flag) {

    if (sl->init_flag == 0) {
        sequence_learner_initialize(sl);
    }

    page_step(sl->input);
    page_step(sl->output);
    page_fetch(sl->input);

    if (sl->input->changed_flag || sl->output->changed_flag) {
        struct ActArray* input_aa = page_get_actarray(sl->input, 0);

        sequence_learner_overlap_(sl, input_aa);
        sequence_learner_activate_(sl, input_aa, learn_flag);
        
        if (learn_flag) {
            sequence_learner_learn_(sl, input_aa);
        }

        page_compute_changed(sl->output);
    }
    else {
        page_copy_previous_to_current(sl->output);
        sl->output->changed_flag = 0;
    }
}

// =============================================================================
// Get Score
// =============================================================================
double sequence_learner_get_score(struct SequenceLearner* sl) {
    return sl->pct_score;
}

// =============================================================================
// Overlap
// =============================================================================
void sequence_learner_overlap_(
        struct SequenceLearner* sl,
        const struct ActArray* input_aa) {

    // loop through each active column
    for (uint32_t k = 0; k < input_aa->num_acts; k++) {
        uint32_t c = input_aa->acts[k];

        for (uint32_t cd = 0; cd < sl->num_dpc; cd++) {
            uint32_t d = cd + (c * sl->num_dpc);

            bitarray_clear(sl->connections_ba);
            
            for (uint32_t r = 0; r < sl->num_rpd; r++) {
                if (sl->coincidence_sets[d].perms[r] > 0) { // TODO: change 0 to perm_thr
                    uint32_t bit = sl->coincidence_sets[d].addrs[r];
                    bitarray_set_bit(sl->connections_ba, bit);
                }
            }

            struct BitArray* output_ba = page_get_bitarray(sl->output, 1);
            bitarray_and(sl->connections_ba, output_ba, sl->activeconns_ba);
            sl->coincidence_sets[d].overlap = bitarray_count(sl->activeconns_ba);
        }
    }
}

// =============================================================================
// Activate
// =============================================================================
void sequence_learner_activate_(
        struct SequenceLearner* sl,
        const struct ActArray* input_aa,
        const uint32_t learn_flag) {

    sl->pct_score = 0.0;

    for (uint32_t k = 0; k < input_aa->num_acts; k++) {
        uint32_t c = input_aa->acts[k];
        uint32_t surprise_flag = 1;

        // handle recognition
        for (uint32_t cd = 0; cd < sl->num_dpc; cd++) {
            uint32_t d = cd + (c * sl->num_dpc);
            
            sl->coincidence_sets[d].state = 0;
            
            if (sl->coincidence_sets[d].overlap >= sl->d_thresh) {
                uint32_t s = d / sl->num_dps;
                sl->coincidence_sets[d].state = 1;
                page_set_bit(sl->output, 0, s);
                surprise_flag = 0;
            }
        }

        // handle surprise
        if (surprise_flag) {
            sl->pct_score++;

            uint32_t s_beg = c * sl->num_spc; // initial statelet
            uint32_t s_end = s_beg + sl->num_spc - 1; // final statelet
            uint32_t s_rand = utils_rand_uint(s_beg, s_end); // random statelet

            // activate random statelet
            page_set_bit(sl->output, 0, s_rand);

            // activate next available coincidence detector
            if (learn_flag) {
                uint32_t d_beg = s_rand * sl->num_dps;
                uint32_t d_next = d_beg + sl->n_next_d[s_rand];                

                sl->coincidence_sets[d_next].state = 1;

                if (sl->n_next_d[s_rand] < sl->num_dps) {
                    sl->n_next_d[s_rand]++;
                }
            }

            // activate all historical statelets
            for (uint32_t s = s_beg; s <= s_end; s++) {
                if (s != s_rand && sl->n_next_d[s] > 0) {
                    page_set_bit(sl->output, 0, s);

                    if (learn_flag) {
                        uint32_t d_beg_ = s * sl->num_dps;
                        uint32_t d_next_ = d_beg_ + sl->n_next_d[s];
                        sl->coincidence_sets[d_next_].state = 1;
                        if (sl->n_next_d[s] < sl->num_dps) {
                            sl->n_next_d[s]++;
                        }
                    }
                }
            }
        }
    }

    sl->pct_score = sl->pct_score / input_aa->num_acts;
}

// =============================================================================
// Learn
// =============================================================================
void sequence_learner_learn_(
        struct SequenceLearner* sl,
        const struct ActArray* input_aa) {

    for (uint32_t k = 0; k < input_aa->num_acts; k++) {
        uint32_t c = input_aa->acts[k];

        for (uint32_t cd = 0; cd < sl->num_dpc; cd++) {
            uint32_t d = cd + (c * sl->num_dpc);

            if (sl->coincidence_sets[d].state == 1) {
                coincidence_set_learn_move(
                    &sl->coincidence_sets[d],
                    page_get_bitarray(sl->output, 1),
                    page_get_actarray(sl->output, 1),
                    sl->perm_inc,
                    sl->perm_dec);
            }
        }
    }
}