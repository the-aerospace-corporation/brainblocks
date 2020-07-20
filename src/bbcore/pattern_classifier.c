#include "pattern_classifier.h"

#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
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
        const double pct_learn) {

    // error check
    if (num_l == 0) {
        perror("Error: PatternClassifier num_l == 0");
        exit(1);
    }

    if (num_s == 0) {
        perror("Error: PatternClassifier num_s == 0");
        exit(1);
    }

    if (num_as == 0) {
        perror("Error: PatternClassifier num_as == 0");
        exit(1);
    }

    if (num_as > num_s) {
        perror("Error: PatternClassifier num_as > num_s");
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        perror("Error: PatternClassifier perm_thr > 99");
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        perror("Error: PatternClassifier perm_inc > 99");
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        perror("Error: PatternClassifier perm_dec > 99");
        exit(1);
    }

    if (pct_pool <= 0.0 || pct_pool > 1.0) {
        perror("Error: PatternClassifier pct_pool must be between 0.0 and 1.0 and greater than 0.0");
        exit(1);
    }

    if (pct_conn < 0.0 || pct_conn > 1.0) {
        perror("Error: PatternClassifier pct_conn must be between 0.0 and 1.0");
        exit(1);
    }

    if (pct_learn < 0.0 || pct_learn > 1.0) {
        perror("Error: PatternClassifier pct_learn must be between 0.0 and 1.0");
        exit(1);
    }

    // initialize variables
    pc->num_l = num_l;
    pc->num_s = num_s;
    pc->num_as = num_as;
    pc->num_spl = (uint32_t)(pc->num_s / pc->num_l);
    pc->perm_thr = perm_thr;
    pc->perm_inc = perm_inc;
    pc->perm_dec = perm_dec;
    pc->pct_pool = pct_pool;
    pc->pct_conn = pct_conn;
    pc->pct_learn = pct_learn;
    pc->init_flag = 0;
    pc->s_labels = NULL;
    pc->learn_mask = NULL;
    pc->labels = malloc(pc->num_l * sizeof(*pc->labels));
    pc->probs = calloc(pc->num_l, sizeof(*pc->probs));    
    pc->input  = malloc(sizeof(*pc->input));
    pc->output = malloc(sizeof(*pc->output));
    pc->coincidence_sets = NULL;

    // construct pages
    page_construct(pc->input, 2, 0);
    page_construct(pc->output, 2, pc->num_s);

    // initialize labels
    for (uint32_t l = 0; l < pc->num_l; l++) {
        pc->labels[l] = labels[l];
    }
}

// =============================================================================
// Destructor
// =============================================================================
void pattern_classifier_destruct(struct PatternClassifier* pc) {

    // cleanup initialized pointers if applicable
    if (pc->init_flag == 1) {
        
        // destruct each element in coincidence_sets
        for (uint32_t s = 0; s < pc->num_s; s++) {
            coincidence_set_destruct(&pc->coincidence_sets[s]);
        }
    }

    // destruct input and output pages
    page_destruct(pc->input);
    page_destruct(pc->output);

    // free pointers
    free(pc->input);
    free(pc->output);
    free(pc->coincidence_sets);
    free(pc->s_labels);
    free(pc->learn_mask);
    free(pc->labels);
    free(pc->probs);
}

// =============================================================================
// Initialize
// =============================================================================
void pattern_classifier_initialize(struct PatternClassifier* pc) {

    // initialize Pages
    page_initialize(pc->input);
    page_initialize(pc->output);

    // construct coincidence_sets
    uint32_t num_i = pc->input->bitarrays[0]->num_bits;
    uint32_t num_spd = (uint32_t)((double)num_i * pc->pct_pool);
    uint32_t num_learns = (uint32_t)((double)num_spd * pc->pct_learn);
    uint32_t num_conn = (uint32_t)((double)num_spd * pc->pct_conn);

    pc->coincidence_sets = malloc(pc->num_s * sizeof(*pc->coincidence_sets));

    for (uint32_t s = 0; s < pc->num_s; s++) {
        coincidence_set_construct_pooled(
            &pc->coincidence_sets[s], num_i, num_spd, num_conn, pc->perm_thr);
    }

    // initialize neuron labels
    pc->s_labels = malloc(pc->num_s * sizeof(*pc->coincidence_sets));

    for (uint32_t s = 0; s < pc->num_s; s++) {
        pc->s_labels[s] = (uint32_t)(s / pc->num_spl);
    }

    // initialize learning mask
    pc->learn_mask = calloc(num_spd, sizeof(*pc->learn_mask));
    for (uint32_t l = 0; l < num_learns; l++) {
        pc->learn_mask[l] = 1;
    }

    // set init_flag to true
    pc->init_flag = 1;
}

// =============================================================================
// Save
// =============================================================================
void pattern_classifier_save(struct PatternClassifier* pc, const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"wb")) == NULL) {
       printf("Error: pattern_classifier_save() cannot open file");
       exit(1);
    }

    for (uint32_t s = 0; s < pc->num_s; s++) {
        struct CoincidenceSet* cs = &pc->coincidence_sets[s];
        fwrite(cs->addrs, cs->num_r * sizeof(uint32_t), 1, fptr);
        fwrite(cs->perms, cs->num_r * sizeof(int32_t), 1, fptr);
    }

    fclose(fptr); 
}

// =============================================================================
// Load
// =============================================================================
void pattern_classifier_load(struct PatternClassifier* pc, const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"rb")) == NULL) {
       printf("Error: pattern_classifier_load() cannot open file\n");
       exit(1);
    }

    for (uint32_t s = 0; s < pc->num_s; s++) {
        struct CoincidenceSet* cs = &pc->coincidence_sets[s];

        if (fread(cs->addrs, cs->num_r * sizeof(uint32_t), 1, fptr) == 0) {
            printf("Error:\n"); // TODO
        }

        if (fread(cs->perms, cs->num_r * sizeof(int32_t), 1, fptr) == 0) {
            printf("Error:\n"); // TODO
        }

        coincidence_set_update_connections(
            &pc->coincidence_sets[s], pc->perm_thr);
    }

    fclose(fptr); 
}

// =============================================================================
// Clear
// =============================================================================
void pattern_classifier_clear(struct PatternClassifier* pc) {
    page_clear_bits(pc->input, 0); // current
    page_clear_bits(pc->input, 1); // previous
    page_clear_bits(pc->output, 0); // current
    page_clear_bits(pc->output, 1); // previous
}

// =============================================================================
// Compute
// =============================================================================
void pattern_classifier_compute(
        struct PatternClassifier* pc,
        const uint32_t in_label,
        const uint32_t learn_flag) {

    if (pc->init_flag == 0) {
        pattern_classifier_initialize(pc);
    }

    page_step(pc->input);
    page_step(pc->output);
    page_fetch(pc->input);

    pattern_classifier_overlap_(pc);
    pattern_classifier_activate_(pc);

    if (learn_flag) {
        pattern_classifier_learn_(pc, in_label);
    }
}

// =============================================================================
// Update Probabilities
// =============================================================================
void pattern_classifier_update_probabilities(struct PatternClassifier* pc) {
    uint32_t acts = 0;

    for (uint32_t l = 0; l < pc->num_l; l++) {
        pc->probs[l] = 0.0;
    }

    for (uint32_t s = 0; s < pc->num_s; s++) {
        if(page_get_bit(pc->output, 0, s)) {
            pc->probs[pc->s_labels[s]]++;
            acts++;
        }
    }        

    if (acts > 0) {
        for (uint32_t l = 0; l < pc->num_l; l++) {
            pc->probs[l] = pc->probs[l] / acts;
        }
    }
}

// =============================================================================
// Overlap
// =============================================================================
void pattern_classifier_overlap_(struct PatternClassifier* pc) {
    struct BitArray* input_ba = page_get_bitarray(pc->input, CURR);
    for (uint32_t s = 0; s < pc->num_s; s++) {
        coincidence_set_overlap(&pc->coincidence_sets[s], input_ba);
    }
}

// =============================================================================
// Activate
// =============================================================================
void pattern_classifier_activate_(struct PatternClassifier* pc) {
    for (uint32_t k = 0; k < pc->num_as; k++) {
        //uint32_t beg_idx = utils_rand_uint(0, pc->num_s); //TODO: figure out random start
        uint32_t beg_idx = 0;
        uint32_t max_val = 0;
        uint32_t max_idx = beg_idx;

        for (uint32_t s = 0; s < pc->num_s; s++) {
            uint32_t j = s + beg_idx;
            uint32_t d = (j < pc->num_s) ? j : (j - pc->num_s);
            
            if (pc->coincidence_sets[d].templap > max_val) {
                max_val = pc->coincidence_sets[d].templap;
                max_idx = d;
            }
        }

        page_set_bit(pc->output, 0, max_idx);
        pc->coincidence_sets[max_idx].templap = 0;
    }
}

// =============================================================================
// Learn
// =============================================================================
void pattern_classifier_learn_(
        struct PatternClassifier* pc,
        const uint32_t in_label) {

    uint32_t has_label = 0;
    uint32_t label_idx = 0;

    for (uint32_t l = 0; l < pc->num_l; l++) {
        if (pc->labels[l] == in_label) {
            has_label = 1;
            label_idx = l;
            break;
        }
    }

    if(has_label) {
        struct BitArray* input_ba = page_get_bitarray(pc->input, CURR);
        struct ActArray* output_aa = page_get_actarray(pc->output, CURR);

        for (uint32_t k = 0; k < output_aa->num_acts; k++) {
            uint32_t d = output_aa->acts[k];
            utils_shuffle(pc->learn_mask, pc->coincidence_sets[d].num_r);

            if (pc->s_labels[d] == in_label) {
                coincidence_set_learn(
                    &pc->coincidence_sets[d],
                    input_ba,
                    pc->learn_mask,
                    pc->perm_inc,
                    pc->perm_dec);
            }
            else {
                coincidence_set_punish(
                    &pc->coincidence_sets[d],
                    input_ba,
                    pc->learn_mask,
                    pc->perm_inc);
            }

            coincidence_set_update_connections(
                &pc->coincidence_sets[d],
                pc->perm_thr);
        }
    }
}