#include "pattern_pooler.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdio>

// =============================================================================
// Constructor
// =============================================================================
PatternPooler::PatternPooler(
    const uint32_t num_s,
    const uint32_t num_as,
    const uint8_t perm_thr,
    const uint8_t perm_inc,
    const uint8_t perm_dec,
    const double pct_pool,
    const double pct_conn,
    const double pct_learn)
{

    // error check
    if (num_s == 0) {
        std::cout << "Error in PatternPooler::PatternPooler(): num_s == 0" << std::endl;
        exit(1);
    }

    if (num_as == 0) {
        std::cout << "Error in PatternPooler::PatternPooler(): num_as == 0" << std::endl;
        exit(1);
    }
    
    if (num_as > num_s) {
        std::cout << "Error in PatternPooler::PatternPooler(): num_as > num_s" << std::endl;
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        std::cout << "Error in PatternPooler::PatternPooler(): perm_thr > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        std::cout << "Error in PatternPooler::PatternPooler(): perm_inc > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        std::cout << "Error in PatternPooler::PatternPooler(): perm_dec > PERM_MAX" << std::endl;
        exit(1);
    }

    if (pct_pool <= 0.0 || pct_pool > 1.0) {
        std::cout << "Error in PatternPooler::PatternPooler(): pct_pool must be between 0.0 and 1.0 and greater than 0.0" << std::endl;
        exit(1);
    }

    if (pct_conn < 0.0 || pct_conn > 1.0) {
        std::cout << "Error in PatternPooler::PatternPooler(): pct_conn must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }

    if (pct_learn < 0.0 || pct_learn > 1.0) {
        std::cout << "Error in PatternPooler::PatternPooler(): pct_learn must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }

    // setup variables
    this->num_s = num_s;
    this->num_as = num_as;
    this->perm_thr = perm_thr;    
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_pool = pct_pool;
    this->pct_conn = pct_conn;
    this->pct_learn = pct_learn;
    this->init_flag = false;

    // setup input page (output and hidden constructed in initialize())
    input.set_num_bitarrays(2);
    input.set_num_bits(0);
}

// =============================================================================
// Initialize
// =============================================================================
void PatternPooler::initialize() {

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

    // set init_flag to true
    this->init_flag = true;
}

// =============================================================================
// Save
// =============================================================================
void PatternPooler::save(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"wb")) == NULL) {
       std::cout << "Error in PatternPooler::save(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        std::cout << "Error in PatternPooler::save(): block not initialized" << std::endl;
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
void PatternPooler::load(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"rb")) == NULL) {
       std::cout << "Error in PatternPooler::load(): cannot open file" << std::endl;
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
// Clear
// =============================================================================
void PatternPooler::clear_states() {
    input[CURR].clear_bits();
    input[PREV].clear_bits();
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}

// =============================================================================
// Compute
// =============================================================================
void PatternPooler::compute(const uint32_t learn_flag) {
    if (init_flag == false) {
        initialize();
    }

    input.step();
    output.step();
    input.fetch();

    if (input.has_changed()) {
        overlap();
        activate();

        if (learn_flag) {
            learn();
        }

        output.compute_changed();
    }
    else {
        output.copy_previous_to_current();
        output.set_changed_flag(false);
    }
}

// =============================================================================
// Overlap
// =============================================================================
void PatternPooler::overlap() {
    for (uint32_t s = 0; s < num_s; s++) {
        d_output_overlaps[s] = d_output[s].overlap(input[CURR], perm_thr);
        d_output_templaps[s] = d_output_overlaps[s];
    }
}

// =============================================================================
// Activate
// =============================================================================
void PatternPooler::activate() {

    // activate the statelets with the k-highest overlap score
    for (uint32_t k = 0; k < num_as; k++) {
        uint32_t beg_idx = utils_rand_uint(0, num_s); // random start location
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
void PatternPooler::learn() {

    // get output activations
    std::vector<uint32_t> output_acts = output[CURR].get_acts();

    // for each active output statelet learn from input
    for (uint32_t k = 0; k < output_acts.size(); k++) {
        uint32_t s = output_acts[k];
        lmask_ba.random_shuffle();
        d_output[s].learn(input[CURR], lmask_ba, perm_inc, perm_dec);
    }
}