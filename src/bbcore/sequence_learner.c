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
    sl->s_next_d = NULL;
    sl->input = malloc(sizeof(*sl->input));
    sl->hidden = malloc(sizeof(*sl->hidden));
    sl->output = malloc(sizeof(*sl->output));
    sl->connections_ba = malloc(sizeof(*sl->connections_ba));
    sl->activeconns_ba = malloc(sizeof(*sl->activeconns_ba));
    sl->d_hidden = NULL;
    sl->d_output = NULL;
    sl->count_hs = 0;
    sl->count_hd = 0;

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
        page_destruct(sl->hidden);
        page_destruct(sl->output);
        bitarray_destruct(sl->connections_ba);
        bitarray_destruct(sl->activeconns_ba);

        // destruct each element in d_hidden and d_output
        for (uint32_t d = 0; d < sl->num_d; d++) {
            coincidence_set_destruct(&sl->d_hidden[d]);
            coincidence_set_destruct(&sl->d_output[d]);
        }
    }

    // destruct input page
    page_destruct(sl->input);

    // free pointers
    free(sl->s_next_d);
    free(sl->input);
    free(sl->hidden);
    free(sl->output);
    free(sl->d_hidden);
    free(sl->d_output);
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

    // construct and initialize output and hidden pages
    page_construct(sl->hidden, 2, sl->num_s);
    page_construct(sl->output, 2, sl->num_s);
    page_initialize(sl->hidden);
    page_initialize(sl->output);

    // construct bitarrays
    bitarray_construct(sl->connections_ba, sl->num_s);
    bitarray_construct(sl->activeconns_ba, sl->num_s);

    // initialize array of statelets' next coincidence set
    sl->s_next_d = calloc(sl->num_s, sizeof(*sl->s_next_d));

    // construct hidden coincidence detectors
    sl->d_hidden = malloc(sl->num_d * sizeof(*sl->d_hidden));
    for (uint32_t d = 0; d < sl->num_d; d++) {
        coincidence_set_construct(&sl->d_hidden[d], sl->num_rpd);
    }

    // construct output coincidence detectors
    sl->d_output = malloc(sl->num_d * sizeof(*sl->d_output));
    for (uint32_t d = 0; d < sl->num_d; d++) {
        coincidence_set_construct(&sl->d_output[d], sl->num_rpd);
    }

    // set init_flag to true
    sl->init_flag = 1;
}

// =============================================================================
// Save
// =============================================================================
void sequence_learner_save(struct SequenceLearner* sl, const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"wb")) == NULL) {
       printf("Error in sequence_learner_save(): cannot open file");
       exit(1);
    }

    // check if block has been initialized
    if (sl->init_flag == 0) {
        printf("Error in sequence_learner_save(): block not initialized\n");
    }

    struct CoincidenceSet* cs;

    // save hidden coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < sl->num_d; d++) {
        cs = &sl->d_hidden[d];
        fwrite(cs->addrs, cs->num_r * sizeof(cs->addrs[0]), 1, fptr);
        fwrite(cs->perms, cs->num_r * sizeof(cs->perms[0]), 1, fptr);
    }
    
    // save output coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < sl->num_d; d++) {
        cs = &sl->d_output[d];
        fwrite(cs->addrs, cs->num_r * sizeof(cs->addrs[0]), 1, fptr);
        fwrite(cs->perms, cs->num_r * sizeof(cs->perms[0]), 1, fptr);        
    }

    // save next available coincidence detector on each statelet
    fwrite(sl->s_next_d, sl->num_s * sizeof(sl->s_next_d[0]), 1, fptr);

    fclose(fptr);
}

// =============================================================================
// Load
// =============================================================================
void sequence_learner_load(struct SequenceLearner* sl, const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"rb")) == NULL) {
       printf("Error in sequence_learner_load(): cannot open file\n");
       exit(1);
    }

    // check if block has been initialized
    if (sl->init_flag == 0) {
        //printf("Error in sequence_learner_load(): block not initialized\n");
        sequence_learner_initialize(sl);
    }

    struct CoincidenceSet* cs;

    // load hidden coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < sl->num_d; d++) {
        cs = &sl->d_hidden[d];
        fread(cs->addrs, cs->num_r * sizeof(cs->addrs[0]), 1, fptr);
        fread(cs->perms, cs->num_r * sizeof(cs->perms[0]), 1, fptr);
    }

    // load output coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < sl->num_d; d++) {
        cs = &sl->d_output[d];
        fread(cs->addrs, cs->num_r * sizeof(cs->addrs[0]), 1, fptr);
        fread(cs->perms, cs->num_r * sizeof(cs->perms[0]), 1, fptr);        
    }

    // load next available coincidence detector on each statelet
    fread(sl->s_next_d, sl->num_s * sizeof(sl->s_next_d[0]), 1, fptr);

    fclose(fptr); 
}

// =============================================================================
// Clear
// =============================================================================
void sequence_learner_clear(struct SequenceLearner* sl) {
    page_clear_bits(sl->input, 0); // current
    page_clear_bits(sl->input, 1); // previous
    page_clear_bits(sl->hidden, 0); // current
    page_clear_bits(sl->hidden, 1); // previous
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
    page_step(sl->hidden);
    page_step(sl->output);
    page_fetch(sl->input);

    if (sl->input->changed_flag || sl->hidden->changed_flag) {
        struct ActArray* input_aa = page_get_actarray(sl->input, 0);

        sequence_learner_overlap(sl, input_aa);
        sequence_learner_activate(sl, input_aa, learn_flag);

        if (learn_flag) {
            sequence_learner_learn(sl, input_aa);
        }

        page_compute_changed(sl->hidden);
        page_compute_changed(sl->output); // TODO: might not be needed since hidden states are the primary driver
    }
    else {
        page_copy_previous_to_current(sl->hidden);
        page_copy_previous_to_current(sl->output);
        sl->hidden->changed_flag = 0;
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
// Get Historical Statelets
// =============================================================================
struct BitArray* sequence_learner_get_historical_statelets(struct SequenceLearner* sl) {
    struct BitArray* hist_ba = malloc(sizeof(*hist_ba)); // TODO: how do I free this?
    bitarray_construct(hist_ba, sl->num_s);
    
    for (uint32_t s = 0; s < sl->num_s; s++) {
        if (sl->s_next_d[s] > 0) {
            bitarray_set_bit(hist_ba, s);
        }
    }
    
    return hist_ba;
}

// =============================================================================
// Overlap
// =============================================================================
void sequence_learner_overlap(
        struct SequenceLearner* sl,
        const struct ActArray* input_aa) {

    // get previous hidden state bitarray
    struct BitArray* hidden_ba = page_get_bitarray(sl->hidden, 1);

    // for every active column
    for (uint32_t k = 0; k < input_aa->num_acts; k++) {
        uint32_t c = input_aa->acts[k];

        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < sl->num_dpc; cd++) {
            uint32_t d = cd + (c * sl->num_dpc);

            // update the connections bitarray with hidden coincidence detector information
            bitarray_clear(sl->connections_ba);
            for (uint32_t r = 0; r < sl->num_rpd; r++) {
                if (sl->d_hidden[d].perms[r] >= sl->perm_thr) {
                    bitarray_set_bit(sl->connections_ba, sl->d_hidden[d].addrs[r]);
                }
            }

            // overlap hidden coincidence detector connections with previous hidden state
            bitarray_and(sl->connections_ba, hidden_ba, sl->activeconns_ba);
            sl->d_hidden[d].overlap = bitarray_count(sl->activeconns_ba);

            // update the connections bitarray with output coincidence detector information
            bitarray_clear(sl->connections_ba);
            for (uint32_t r = 0; r < sl->num_rpd; r++) {
                if (sl->d_output[d].perms[r] >= sl->perm_thr) {
                    bitarray_set_bit(sl->connections_ba, sl->d_output[d].addrs[r]);
                }
            }

            // overlap output coincidence detector connections with previous hidden state
            bitarray_and(sl->connections_ba, hidden_ba, sl->activeconns_ba);
            sl->d_output[d].overlap = bitarray_count(sl->activeconns_ba);
        }
    }
}

// =============================================================================
// Activate
// =============================================================================
void sequence_learner_activate(
        struct SequenceLearner* sl,
        const struct ActArray* input_aa,
        const uint32_t learn_flag) {

    sl->pct_score = 0.0;

    // for every active column
    for (uint32_t k = 0; k < input_aa->num_acts; k++) {
        uint32_t c = input_aa->acts[k];
        uint32_t hidden_surprise_flag = 1;
        uint32_t output_surprise_flag = 1;

        // ====================
        // Recognition
        // ====================
        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < sl->num_dpc; cd++) {

            // get global index of the coincidence detector
            uint32_t d = cd + (c * sl->num_dpc); // global index of coincidence detector

            // deactivate hidden and output coincidence detectors
            sl->d_hidden[d].state = 0;
            sl->d_output[d].state = 0;

            // if hidden coincidence detector overlap is above the threshold
            if (sl->d_hidden[d].overlap >= sl->d_thresh) {
                uint32_t s = d / sl->num_dps;   // get global index of statelet
                sl->d_hidden[d].state = 1;      // activate hidden coincidence detector
                page_set_bit(sl->hidden, 0, s); // activate hidden statelet
                hidden_surprise_flag = 0;
            }

            // if output coincidence detector overlap is above the threshold
            if (sl->d_output[d].overlap >= sl->d_thresh) {
                uint32_t s = d / sl->num_dps;   // get global index of statelet
                //sl->d_hidden[d].state = 1;      // activate output coincidence detector
                page_set_bit(sl->output, 0, s); // activate output statelet
                output_surprise_flag = 0;
            }
        }

        /*
        // handles instances where hidden states closed loops with historical statelets
        // but output statelets never empirically learned the transition
        if (hidden_surprise_flag == 0 && output_surprise_flag == 1) {
            sl->pct_score++;
            
            uint32_t s_beg = c * sl->num_spc;                // global index of first statelet 
            uint32_t s_end = s_beg + sl->num_spc - 1;        // global index of final statelet 
            uint32_t s_rand = utils_rand_uint(s_beg, s_end); // global index of random statelet

            // activate random output statelets
            page_set_bit(sl->output, 0, s_rand);

            // activate next available output coincidence detector
            if (learn_flag) {
                uint32_t d_beg = s_rand * sl->num_dps;          // global index of first coincidence detector on random statelet
                uint32_t d_next = d_beg + sl->s_next_d[s_rand]; // global index of next available coincidence detector on random statelet

                // activate next available hidden and output coincidence detector
                sl->d_output[d_next].state = 1;

                // if next available coincidence detector is less than the number of coincidence detectors per statelet
                if (sl->s_next_d[s_rand] < sl->num_dps) {

                    // update historical coincidence detector and statelet counters
                    // remember: both hidden and output were activated in this case
                    sl->count_hd += 1;
                    if (sl->s_next_d[s_rand] == 0) {
                        sl->count_hs += 1;
                    }
                    
                    // update next available coincidence detector on the random statelet
                    sl->s_next_d[s_rand]++;
                }
            }
        }
        */

        // ====================
        // Surprise
        // ====================
        if (hidden_surprise_flag == 1) {
            sl->pct_score++;

            uint32_t s_beg = c * sl->num_spc;                // global index of first statelet 
            uint32_t s_end = s_beg + sl->num_spc - 1;        // global index of final statelet 
            uint32_t s_rand = utils_rand_uint(s_beg, s_end); // global index of random statelet

            // activate random hidden and output statelets
            page_set_bit(sl->hidden, 0, s_rand);
            page_set_bit(sl->output, 0, s_rand);

            // activate next available hidden and output coincidence detectors
            if (learn_flag) {
                uint32_t d_beg = s_rand * sl->num_dps;          // global index of first coincidence detector on random statelet
                uint32_t d_next = d_beg + sl->s_next_d[s_rand]; // global index of next available coincidence detector on random statelet

                // activate next available hidden and output coincidence detector
                sl->d_hidden[d_next].state = 1;
                sl->d_output[d_next].state = 1;

                // if next available coincidence detector is less than the number of coincidence detectors per statelet
                if (sl->s_next_d[s_rand] < sl->num_dps-1) {

                    // update historical coincidence detector and statelet counters
                    // remember: both hidden and output were activated in this case
                    sl->count_hd += 2; 
                    if (sl->s_next_d[s_rand] == 0) {
                        sl->count_hs += 2;
                    }
                    
                    // update next available coincidence detector on the random statelet
                    sl->s_next_d[s_rand]++;
                }
            }

            // activate all hidden historical statelets
            // for each global index of statelets on the column
            for (uint32_t s = s_beg; s <= s_end; s++) {

                // if the statelet is not the random statelet
                // and the statelet has at lease one coincidence detector
                if (s != s_rand && sl->s_next_d[s] > 0) {

                    // activate the hidden statlet
                    page_set_bit(sl->hidden, 0, s);

                    // activate next available hidden coincidence detector
                    if (learn_flag) {
                        uint32_t d_beg = s * sl->num_dps;
                        uint32_t d_next = d_beg + sl->s_next_d[s];

                        // activate next available hidden coincidence detector
                        sl->d_hidden[d_next].state = 1;

                        if (sl->s_next_d[s] < sl->num_dps-1) {
                            sl->count_hd++;
                            if (sl->s_next_d[s] == 0) {
                                sl->count_hs++;
                            }

                            sl->s_next_d[s]++;
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
void sequence_learner_learn(
        struct SequenceLearner* sl,
        const struct ActArray* input_aa) {

    // for every active column
    for (uint32_t k = 0; k < input_aa->num_acts; k++) {
        uint32_t c = input_aa->acts[k];

        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < sl->num_dpc; cd++) {
            
            // get global index of the coincidence detector
            uint32_t d = cd + (c * sl->num_dpc);

            // learn active hidden coincidence set
            if (sl->d_hidden[d].state == 1) {
                coincidence_set_learn_move(
                    &sl->d_hidden[d],
                    page_get_bitarray(sl->hidden, 1),
                    page_get_actarray(sl->hidden, 1),
                    sl->perm_inc,
                    sl->perm_dec);
            }

            // learn active output coincidence set
            if (sl->d_output[d].state == 1) {
                coincidence_set_learn_move(
                    &sl->d_output[d],
                    page_get_bitarray(sl->output, 1),
                    page_get_actarray(sl->output, 1),
                    sl->perm_inc,
                    sl->perm_dec);
            }
        }
    }
}