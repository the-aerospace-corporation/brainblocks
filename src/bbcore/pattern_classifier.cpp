#include "pattern_classifier.hpp"
#include "utils.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
PatternClassifier::PatternClassifier(
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
    this->num_l = num_l;
    this->num_s = num_s;
    this->num_as = num_as;
    this->num_spl = (uint32_t)(this->num_s / this->num_l);
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_pool = pct_pool;
    this->pct_conn = pct_conn;
    this->pct_learn = pct_learn;
    this->init_flag = 0;
    this->s_labels = NULL;
    this->learn_mask = NULL;
    this->labels = (uint32_t*)malloc(this->num_l * sizeof(*this->labels));
    this->probs = (double*)calloc(this->num_l, sizeof(*this->probs));    
    this->input  = (Page*)malloc(sizeof(*this->input));
    this->output = (Page*)malloc(sizeof(*this->output));
    this->coincidence_sets = NULL;

    // construct pages
    page_construct(this->input, 2, 0);
    page_construct(this->output, 2, this->num_s);

    // initialize labels
    for (uint32_t l = 0; l < this->num_l; l++) {
        this->labels[l] = labels[l];
    }
}

// =============================================================================
// Destructor
// =============================================================================
PatternClassifier::~PatternClassifier() {

    // cleanup initialized pointers if applicable
    if (this->init_flag == 1) {
        
        // destruct each element in coincidence_sets
        for (uint32_t s = 0; s < this->num_s; s++) {
            coincidence_set_destruct(&this->coincidence_sets[s]);
        }
    }

    // destruct input and output pages
    page_destruct(this->input);
    page_destruct(this->output);

    // free pointers
    free(this->input);
    free(this->output);
    free(this->coincidence_sets);
    free(this->s_labels);
    free(this->learn_mask);
    free(this->labels);
    free(this->probs);
}

// =============================================================================
// Initialize
// =============================================================================
void PatternClassifier::initialize() {

    // initialize Pages
    page_initialize(this->input);
    page_initialize(this->output);

    // construct coincidence_sets
    uint32_t num_i = this->input->bitarrays[0]->num_bits;
    uint32_t num_spd = (uint32_t)((double)num_i * this->pct_pool);
    uint32_t num_learns = (uint32_t)((double)num_spd * this->pct_learn);
    uint32_t num_conn = (uint32_t)((double)num_spd * this->pct_conn);

    this->coincidence_sets = (CoincidenceSet*)malloc(this->num_s * sizeof(*this->coincidence_sets));

    for (uint32_t s = 0; s < this->num_s; s++) {
        coincidence_set_construct_pooled(
            &this->coincidence_sets[s], num_i, num_spd, num_conn, this->perm_thr);
    }

    // initialize neuron labels
    this->s_labels = (uint32_t*)malloc(this->num_s * sizeof(*this->coincidence_sets));

    for (uint32_t s = 0; s < this->num_s; s++) {
        this->s_labels[s] = (uint32_t)(s / this->num_spl);
    }

    // initialize learning mask
    this->learn_mask = (uint32_t*)calloc(num_spd, sizeof(*this->learn_mask));
    for (uint32_t l = 0; l < num_learns; l++) {
        this->learn_mask[l] = 1;
    }

    // set init_flag to true
    this->init_flag = 1;
}

// =============================================================================
// Save
// =============================================================================
void PatternClassifier::save(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"wb")) == NULL) {
       printf("Error: pattern_classifier_save() cannot open file");
       exit(1);
    }

    for (uint32_t s = 0; s < this->num_s; s++) {
        struct CoincidenceSet* cs = &this->coincidence_sets[s];
        fwrite(cs->addrs, cs->num_r * sizeof(uint32_t), 1, fptr);
        fwrite(cs->perms, cs->num_r * sizeof(int32_t), 1, fptr);
    }

    fclose(fptr); 
}

// =============================================================================
// Load
// =============================================================================
void PatternClassifier::load(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"rb")) == NULL) {
       printf("Error: pattern_classifier_load() cannot open file\n");
       exit(1);
    }

    for (uint32_t s = 0; s < this->num_s; s++) {
        struct CoincidenceSet* cs = &this->coincidence_sets[s];

        if (fread(cs->addrs, cs->num_r * sizeof(uint32_t), 1, fptr) == 0) {
            printf("Error:\n"); // TODO
        }

        if (fread(cs->perms, cs->num_r * sizeof(int32_t), 1, fptr) == 0) {
            printf("Error:\n"); // TODO
        }

        coincidence_set_update_connections(
            &this->coincidence_sets[s], this->perm_thr);
    }

    fclose(fptr); 
}

// =============================================================================
// Clear
// =============================================================================
// TODO: make clear_states()
void PatternClassifier::clear() {
    page_clear_bits(this->input, 0); // current
    page_clear_bits(this->input, 1); // previous
    page_clear_bits(this->output, 0); // current
    page_clear_bits(this->output, 1); // previous
}

// =============================================================================
// Compute
// =============================================================================
void PatternClassifier::compute(const uint32_t in_label, const uint32_t learn_flag) {

    if (this->init_flag == 0) {
        this->initialize();
    }

    page_step(this->input);
    page_step(this->output);
    page_fetch(this->input);

    this->overlap();
    this->activate();

    if (learn_flag) {
        this->learn(in_label);
    }
}

// =============================================================================
// Update Probabilities
// =============================================================================
void PatternClassifier::update_probabilities() {
    uint32_t acts = 0;

    for (uint32_t l = 0; l < this->num_l; l++) {
        this->probs[l] = 0.0;
    }

    for (uint32_t s = 0; s < this->num_s; s++) {
        if(page_get_bit(this->output, 0, s)) {
            this->probs[this->s_labels[s]]++;
            acts++;
        }
    }        

    if (acts > 0) {
        for (uint32_t l = 0; l < this->num_l; l++) {
            this->probs[l] = this->probs[l] / acts;
        }
    }
}

// =============================================================================
// Overlap
// =============================================================================
void PatternClassifier::overlap() {
    struct BitArray* input_ba = page_get_bitarray(this->input, CURR);
    for (uint32_t s = 0; s < this->num_s; s++) {
        coincidence_set_overlap(&this->coincidence_sets[s], input_ba);
    }
}

// =============================================================================
// Activate
// =============================================================================
void PatternClassifier::activate() {
    for (uint32_t k = 0; k < this->num_as; k++) {
        //uint32_t beg_idx = utils_rand_uint(0, this->num_s); //TODO: figure out random start
        uint32_t beg_idx = 0;
        uint32_t max_val = 0;
        uint32_t max_idx = beg_idx;

        for (uint32_t s = 0; s < this->num_s; s++) {
            uint32_t j = s + beg_idx;
            uint32_t d = (j < this->num_s) ? j : (j - this->num_s);
            
            if (this->coincidence_sets[d].templap > max_val) {
                max_val = this->coincidence_sets[d].templap;
                max_idx = d;
            }
        }

        page_set_bit(this->output, 0, max_idx);
        this->coincidence_sets[max_idx].templap = 0;
    }
}

// =============================================================================
// Learn
// =============================================================================
void PatternClassifier::learn(const uint32_t in_label) {

    uint32_t has_label = 0;
    uint32_t label_idx = 0;

    for (uint32_t l = 0; l < this->num_l; l++) {
        if (this->labels[l] == in_label) {
            has_label = 1;
            label_idx = l;
            break;
        }
    }

    if(has_label) {
        struct BitArray* input_ba = page_get_bitarray(this->input, CURR);
        struct ActArray* output_aa = page_get_actarray(this->output, CURR);

        for (uint32_t k = 0; k < output_aa->num_acts; k++) {
            uint32_t d = output_aa->acts[k];
            utils_shuffle(this->learn_mask, this->coincidence_sets[d].num_r);

            if (this->s_labels[d] == in_label) {
                coincidence_set_learn(
                    &this->coincidence_sets[d],
                    input_ba,
                    this->learn_mask,
                    this->perm_inc,
                    this->perm_dec);
            }
            else {
                coincidence_set_punish(
                    &this->coincidence_sets[d],
                    input_ba,
                    this->learn_mask,
                    this->perm_inc);
            }

            coincidence_set_update_connections(
                &this->coincidence_sets[d],
                this->perm_thr);
        }
    }
}

struct BitArray* PatternClassifier::decode() {
    struct ActArray* output_aa = page_get_actarray(this->output, CURR);
    struct BitArray* backtrace_ba = (BitArray*)malloc(sizeof(*backtrace_ba)); // TODO: how do I clean this up? put in parameters instead of return
    uint32_t num_bits = this->coincidence_sets[0].connections_ba->num_bits;
    
    bitarray_construct(backtrace_ba, num_bits);

    for (uint32_t k = 0; k < this->num_as; k++) {
        uint32_t s = output_aa->acts[k];
        struct BitArray* conn_ba = coincidence_set_get_connections(&this->coincidence_sets[s]);
        bitarray_or(conn_ba, backtrace_ba, backtrace_ba);
    }
    
    return backtrace_ba;
}