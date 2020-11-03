#include "pattern_classifier.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdio>

// =============================================================================
// Constructor
// =============================================================================
PatternClassifier::PatternClassifier(
        const std::vector<uint32_t> labels,
        const uint32_t num_s,
        const uint32_t num_as,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec,
        const double pct_pool,
        const double pct_conn,
        const double pct_learn) {

    // error check
    if (labels.size() == 0) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): labels is empty" << std::endl;
        exit(1);
    }

    if (num_s == 0) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): num_s == 0" << std::endl;
        exit(1);
    }

    if (num_as == 0) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): num_as == 0" << std::endl;
        exit(1);
    }

    if (num_as > num_s) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): num_as > num_s" << std::endl;
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): perm_thr > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): perm_inc > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): perm_dec > PERM_MAX" << std::endl;
        exit(1);
    }

    if (pct_pool <= 0.0 || pct_pool > 1.0) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): pct_pool must be between 0.0 and 1.0 and greater than 0.0" << std::endl;
        exit(1);
    }

    if (pct_conn < 0.0 || pct_conn > 1.0) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): pct_conn must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }

    if (pct_learn < 0.0 || pct_learn > 1.0) {
        std::cout << "Error in PatternClassifier::PatternClassifier(): pct_learn must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }

    // setup variables
    this->num_l = (uint32_t)labels.size();
    this->num_s = num_s;
    this->num_as = num_as;
    this->num_spl = (uint32_t)(num_s / num_l);
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_pool = pct_pool;
    this->pct_conn = pct_conn;
    this->pct_learn = pct_learn;
    this->init_flag = false;

    // setup labels and probabilities vectors
    this->labels.resize(num_l);
    this->probs.resize(num_l);
    
    for (uint32_t l = 0; l < num_l; l++) {
        this->labels[l] = labels[l];
    }

    // setup input page (output and hidden constructed in initialize())
    input.set_num_bitarrays(2);
    input.set_num_bits(0);
}

// =============================================================================
// Initialize
// =============================================================================
void PatternClassifier::initialize() {

    // setup and initialize pages
    input.initialize();
    output.set_num_bitarrays(2);
    output.set_num_bits(num_s);
    output.initialize();

    // setup local variables
    uint32_t num_i = input[CURR].get_num_bits();
    uint32_t num_rpd = (uint32_t)((double)num_i * pct_pool);
    uint32_t num_conn = (uint32_t)((double)num_rpd * pct_conn);

    // setup learning mask bitarray
    lmask_ba.resize(num_rpd);
    lmask_ba.random_fill(pct_learn);

    // setup coincidence_sets
    d_output.resize(num_s);
    d_output_overlaps.resize(num_s);
    d_output_templaps.resize(num_s);

    for (uint32_t s = 0; s < num_s; s++) {
        d_output[s].initialize_pool(num_rpd, num_i, num_conn, perm_thr);
    }

    // setup labels for each statelet
    s_labels.resize(num_s);

    for (uint32_t s = 0; s < num_s; s++) {
        s_labels[s] = (uint32_t)(s / num_spl);
    }

    // set init_flag to true
    this->init_flag = true;
}

// =============================================================================
// Save
// =============================================================================
void PatternClassifier::save(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"wb")) == NULL) {
       std::cout << "Error in PatternClassifier::save(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        std::cout << "Error in PatternClassifier::save(): block not initialized" << std::endl;
    }

    CoincidenceSet* cs;

    for (uint32_t s = 0; s < this->num_s; s++) {
        cs = &d_output[s];
        fwrite(cs->get_addrs().data(), cs->get_num_r() * sizeof(cs->get_addrs()[0]), 1, fptr);
        fwrite(cs->get_perms().data(), cs->get_num_r() * sizeof(cs->get_perms()[0]), 1, fptr);
    }

    fclose(fptr); 
}

// =============================================================================
// Load
// =============================================================================
void PatternClassifier::load(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"rb")) == NULL) {
       std::cout << "Error in PatternClassifier::load(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        initialize();
    }

    CoincidenceSet* cs;

    for (uint32_t s = 0; s < this->num_s; s++) {
        cs = &d_output[s];
        fread(cs->get_addrs().data(), cs->get_num_r() * sizeof(cs->get_addrs()[0]), 1, fptr);
        fread(cs->get_perms().data(), cs->get_num_r() * sizeof(cs->get_perms()[0]), 1, fptr);
    }

    fclose(fptr); 
}


// =============================================================================
// Clear States
// =============================================================================
void PatternClassifier::clear_states() {
    input[CURR].clear_bits();
    input[PREV].clear_bits();
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}

// =============================================================================
// Compute
// =============================================================================
void PatternClassifier::compute(const uint32_t label, const uint32_t learn_flag) {
    if (init_flag == false) {
        initialize();
    }

    input.step();
    output.step();
    input.fetch();

    overlap();
    activate();

    if (learn_flag) {
        learn(label);
    }
}

// =============================================================================
// Get Probabilities
// =============================================================================
std::vector<double> PatternClassifier::get_probabilities() {
    uint32_t acts = 0;

    // clear probabilities
    for (uint32_t l = 0; l < num_l; l++) {
        probs[l] = 0.0;
    }

    // increment respective label probability if output statelet is active
    for (uint32_t s = 0; s < num_s; s++) {
        if(output[CURR].get_bit(s) == 1) {
            probs[s_labels[s]]++;
            acts++;
        }
    }        

    // normalize probabilities
    if (acts > 0) {
        for (uint32_t l = 0; l < num_l; l++) {
            probs[l] = probs[l] / (double)acts;
        }
    }

    return probs;
}

// =============================================================================
// Overlap
// =============================================================================
void PatternClassifier::overlap() {
    for (uint32_t s = 0; s < num_s; s++) {
        d_output_overlaps[s] = d_output[s].overlap(input[CURR], perm_thr);
        d_output_templaps[s] = d_output_overlaps[s];
    }
}

// =============================================================================
// Activate
// =============================================================================
void PatternClassifier::activate() {

    // activate the statelets with the k-highest overlap score
    for (uint32_t k = 0; k < num_as; k++) {
        //uint32_t beg_idx = utils_rand_uint(0, num_s - 1); // random start location
        uint32_t beg_idx = 0;
        uint32_t max_val = 0;
        uint32_t max_idx = beg_idx;

        // loop through output statelets
        for (uint32_t i = 0; i < num_s; i++) {

            // account for random start location
            uint32_t j = i + beg_idx;
            uint32_t s = (j < num_s) ? j : (j - num_s);

            // find the statelet with the highest overlap score
            if (d_output_templaps[s] > max_val) {
                max_val = d_output_templaps[s];
                max_idx = s;
            }
        }

        // activate statelet with highest overlap score
        output[CURR].set_bit(max_idx, 1);
        
        // clear highest overlap score in temporary overlap vector
        d_output_templaps[max_idx] = 0;
    }
}

// =============================================================================
// Learn
// =============================================================================
void PatternClassifier::learn(const uint32_t label) {

    bool has_label = false;
    uint32_t label_idx = 0;

    // check if input label exists in known labels
    for (uint32_t l = 0; l < num_l; l++) {
        if (labels[l] == label) {
            has_label = true;
            label_idx = l;
            break;
        }
    }

    // if input label exists in known labels
    if(has_label) {

        // get output activations
        std::vector<uint32_t> output_acts = output[CURR].get_acts();

        // for each active output statelet
        for (uint32_t k = 0; k < output_acts.size(); k++) {
            uint32_t s = output_acts[k];

            // shuffle learning mask
            lmask_ba.random_shuffle();

            // if input label matches statelet label then learn
            if (s_labels[s] == label) {
                d_output[s].learn(input[CURR], lmask_ba, perm_inc, perm_dec);
            }
            
            // if input label does not match statelet label then punish
            else {
                d_output[s].punish(input[CURR], lmask_ba, perm_inc);
            }
        }
    }
    else {
        std::cout << "Warning in PatternClassifier::learn(): inputted label not in known labels" << std::endl;
    }
}

/*
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
*/