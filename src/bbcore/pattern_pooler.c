#include "pattern_pooler.h"

#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
void pattern_pooler_construct(
    struct PatternPooler* pp,
    const uint32_t num_s,
    const uint32_t num_as,
    const uint32_t perm_thr,
    const uint32_t perm_inc,
    const uint32_t perm_dec,
    const double pct_pool,
    const double pct_conn,
    const double pct_learn)
{

    // error check
    if (num_s == 0) {
        perror("Error: PatternPooler num_s == 0");
        exit(1);
    }

    if (num_as == 0) {
        perror("Error: PatternPooler num_as == 0");
        exit(1);
    }
    
    if (num_as > num_s) {
        perror("Error: PatternPooler num_as > num_s");
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        perror("Error: PatternPooler perm_thr > 99");
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        perror("Error: PatternPooler perm_inc > 99");
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        perror("Error: PatternPooler perm_dec > 99");
        exit(1);
    }

    if (pct_pool <= 0.0 || pct_pool > 1.0) {
        perror("Error: PatternPooler pct_pool must be between 0.0 and 1.0 and greater than 0.0");
        exit(1);
    }

    if (pct_conn < 0.0 || pct_conn > 1.0) {
        perror("Error: PatternPooler pct_conn must be between 0.0 and 1.0");
        exit(1);
    }

    if (pct_learn < 0.0 || pct_learn > 1.0) {
        perror("Error: PatternPooler pct_learn must be between 0.0 and 1.0");
        exit(1);
    }

    // initialize variables
    pp->num_s = num_s;
    pp->num_as = num_as;
    pp->perm_thr = perm_thr;    
    pp->perm_inc = perm_inc;
    pp->perm_dec = perm_dec;
    pp->pct_pool = pct_pool;
    pp->pct_conn = pct_conn;
    pp->pct_learn = pct_learn;
    pp->init_flag = 0;
    pp->learn_mask = NULL;
    pp->input  = malloc(sizeof(*pp->input));
    pp->output = malloc(sizeof(*pp->output));
    pp->coincidence_sets = NULL;

    // contruct pages
    page_construct(pp->input, 2, 0);
    page_construct(pp->output, 2, num_s);
}

// =============================================================================
// Destructor
// =============================================================================
void pattern_pooler_destruct(struct PatternPooler* pp) {

    // cleanup initialized pointers if applicable
    if (pp->init_flag == 1) {
        
        // destruct each element in coincidence_sets
        for (uint32_t s = 0; s < pp->num_s; s++) {
            coincidence_set_destruct(&pp->coincidence_sets[s]);
        }
    }

    // destruct input and output pages
    page_destruct(pp->input);
    page_destruct(pp->output);

    // free pointers
    free(pp->input);
    free(pp->output);
    free(pp->coincidence_sets);
    free(pp->learn_mask);
}

// =============================================================================
// Initialize
// =============================================================================
void pattern_pooler_initialize(struct PatternPooler* pp) {

    // initialize Pages
    page_initialize(pp->input);
    page_initialize(pp->output);

    // construct coincidence_sets
    uint32_t num_i = pp->input->bitarrays[0]->num_bits;
    uint32_t num_spd = (uint32_t)((double)num_i * pp->pct_pool);
    uint32_t num_learns = (uint32_t)((double)num_spd * pp->pct_learn);
    uint32_t num_conn = (uint32_t)((double)num_spd * pp->pct_conn);

    pp->coincidence_sets = malloc(pp->num_s * sizeof(*pp->coincidence_sets));

    for (uint32_t s = 0; s < pp->num_s; s++) {
        coincidence_set_construct_pooled(
            &pp->coincidence_sets[s], num_i, num_spd, num_conn, pp->perm_thr);
    }

    // initialize learning mask
    pp->learn_mask = calloc(num_spd, sizeof(*pp->learn_mask));

    for (uint32_t l = 0; l < num_learns; l++) {
        pp->learn_mask[l] = 1;
    }

    // set init_flag to true
    pp->init_flag = 1;
}

// =============================================================================
// Save
// =============================================================================
void pattern_pooler_save(struct PatternPooler* pp, const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"wb")) == NULL) {
       printf("Error in pattern_pooler_save(): cannot open file");
       exit(1);
    }

    // check if block has been initialized
    if (pp->init_flag == 0) {
        printf("Error in pattern_pooler_save(): block not initialized\n");
    }

    // save coincidence detector receptor addresses and permanences
    struct CoincidenceSet* cs;
    for (uint32_t s = 0; s < pp->num_s; s++) {
        cs = &pp->coincidence_sets[s];
        fwrite(cs->addrs, cs->num_r * sizeof(cs->addrs[0]), 1, fptr);
        fwrite(cs->perms, cs->num_r * sizeof(cs->perms[0]), 1, fptr);
    }

    fclose(fptr); 
}

// =============================================================================
// Load
// =============================================================================
void pattern_pooler_load(struct PatternPooler* pp, const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"rb")) == NULL) {
       printf("Error in pattern_pooler_load(): cannot open file\n");
       exit(1);
    }

    // check if block has been initialized
    if (pp->init_flag == 0) {
        printf("Error in pattern_pooler_load(): block not initialized\n");
    }

    // load coincidence detector receptor addresses and permanences
    struct CoincidenceSet* cs;
    for (uint32_t s = 0; s < pp->num_s; s++) {
        cs = &pp->coincidence_sets[s];
        fread(cs->addrs, cs->num_r * sizeof(cs->addrs[0]), 1, fptr);
        fread(cs->perms, cs->num_r * sizeof(cs->perms[0]), 1, fptr);

        coincidence_set_update_connections(
            &pp->coincidence_sets[s], pp->perm_thr);
    }

    fclose(fptr); 
}

// =============================================================================
// Clear
// =============================================================================
void pattern_pooler_clear(struct PatternPooler* pp) {
    page_clear_bits(pp->input, 0); // current
    page_clear_bits(pp->input, 1); // previous
    page_clear_bits(pp->output, 0); // current
    page_clear_bits(pp->output, 1); // previous
}

// =============================================================================
// Compute
// =============================================================================
void pattern_pooler_compute(
        struct PatternPooler* pp,
        const uint32_t learn_flag) {

    if (pp->init_flag == 0) {
        pattern_pooler_initialize(pp);
    }

    page_step(pp->input);
    page_step(pp->output);
    page_fetch(pp->input);

    if (pp->input->changed_flag) {
        pattern_pooler_overlap_(pp);
        pattern_pooler_activate_(pp);

        if (learn_flag) {
            pattern_pooler_learn_(pp);
        }

        page_compute_changed(pp->output);
    }
    else {
        page_copy_previous_to_current(pp->output);
        pp->output->changed_flag = 0;
    }
}

// =============================================================================
// Overlap
// =============================================================================
void pattern_pooler_overlap_(struct PatternPooler* pp) {
    struct BitArray* input_ba = page_get_bitarray(pp->input, CURR);
    for (uint32_t s = 0; s < pp->num_s; s++) {
        coincidence_set_overlap(&pp->coincidence_sets[s], input_ba);
    }
}

// =============================================================================
// Activate
// =============================================================================
void pattern_pooler_activate_(struct PatternPooler* pp) {
    for (uint32_t k = 0; k < pp->num_as; k++) {
        //uint32_t beg_idx = utils_rand_uint(0, pp->num_s);  //TODO: figure out random start
        uint32_t beg_idx = 0;
        uint32_t max_val = 0;
        uint32_t max_idx = beg_idx;

        for (uint32_t s = 0; s < pp->num_s; s++) {
            uint32_t j = s + beg_idx;
            uint32_t d = (j < pp->num_s) ? j : (j - pp->num_s);
            
            if (pp->coincidence_sets[d].templap > max_val) {
                max_val = pp->coincidence_sets[d].templap;
                max_idx = d;
            }
        }

        page_set_bit(pp->output, 0, max_idx);
        pp->coincidence_sets[max_idx].templap = 0;
    }
}

// =============================================================================
// Learn
// =============================================================================
void pattern_pooler_learn_(struct PatternPooler* pp) {
    struct BitArray* input_ba = page_get_bitarray(pp->input, CURR);
    struct ActArray* output_aa = page_get_actarray(pp->output, CURR);

    for (uint32_t k = 0; k < output_aa->num_acts; k++) {
        uint32_t d = output_aa->acts[k];
        utils_shuffle(pp->learn_mask, pp->coincidence_sets[d].num_r);

        coincidence_set_learn(
            &pp->coincidence_sets[d],
            input_ba,
            pp->learn_mask,
            pp->perm_inc,
            pp->perm_dec);

        coincidence_set_update_connections(
            &pp->coincidence_sets[d],
            pp->perm_thr);
    }
}