#include "pattern_classifier_dynamic.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdio>

// =============================================================================
// Constructor
// =============================================================================
PatternClassifierDynamic::PatternClassifierDynamic(
        const uint32_t num_s,
        const uint32_t num_as,
        const uint32_t num_spl,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec,
        const double pct_pool,
        const double pct_conn,
        const double pct_learn) {

    // error check
    if (num_s == 0) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): num_s == 0" << std::endl;
        exit(1);
    }

    if (num_as == 0) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): num_as == 0" << std::endl;
        exit(1);
    }

    if (num_as > num_s) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): num_as > num_s" << std::endl;
        exit(1);
    }


    if (num_spl == 0) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): num_spl == 0" << std::endl;
        exit(1);
    }

    if (num_spl > num_s) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): num_spl > num_s" << std::endl;
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): perm_thr > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): perm_inc > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): perm_dec > PERM_MAX" << std::endl;
        exit(1);
    }

    if (pct_pool <= 0.0 || pct_pool > 1.0) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): pct_pool must be between 0.0 and 1.0 and greater than 0.0" << std::endl;
        exit(1);
    }

    if (pct_conn < 0.0 || pct_conn > 1.0) {
        std::cout << "Error in PatternClassifierDynamic::PatternClassifierDynamic(): pct_conn must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }

    // setup variables
    this->num_s = num_s;
    this->num_as = num_as;
    this->num_spl = num_spl;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_pool = pct_pool;
    this->pct_conn = pct_conn;
    this->pct_learn = pct_learn;
    this->pct_score = 0.0;
    this->init_flag = false;

    // setup input page (output and hidden constructed in initialize())
    input.set_num_bitarrays(2);
    input.set_num_bits(0);
}

// =============================================================================
// Initialize
// =============================================================================
void PatternClassifierDynamic::initialize() {

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
void PatternClassifierDynamic::save(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"wb")) == NULL) {
       std::cout << "Error in PatternClassifierDynamic::save(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        std::cout << "Error in PatternClassifierDynamic::save(): block not initialized" << std::endl;
    }

    CoincidenceSet* cs;

    for (uint32_t s = 0; s < this->num_s; s++) {
        cs = &d_output[s];
        fwrite(cs->get_addrs().data(), cs->get_num_r() * sizeof(cs->get_addrs()[0]), 1, fptr);
        fwrite(cs->get_perms().data(), cs->get_num_r() * sizeof(cs->get_perms()[0]), 1, fptr);
    }

    // TODO: need to save label states

    fclose(fptr); 
}

// =============================================================================
// Load
// =============================================================================
void PatternClassifierDynamic::load(const char* file) {
    FILE *fptr;

    if ((fptr = fopen(file,"rb")) == NULL) {
       std::cout << "Error in PatternClassifierDynamic::load(): cannot open file" << std::endl;
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

    // TODO: need to load label states

    fclose(fptr); 
}


// =============================================================================
// Clear States
// =============================================================================
void PatternClassifierDynamic::clear_states() {
    input[CURR].clear_bits();
    input[PREV].clear_bits();
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}

// =============================================================================
// Compute
// =============================================================================
void PatternClassifierDynamic::compute(const uint32_t label, const uint32_t learn_flag) {
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
// Overlap
// =============================================================================
void PatternClassifierDynamic::overlap() {
    for (uint32_t s = 0; s < num_s; s++) {
        d_output_overlaps[s] = d_output[s].overlap(input[CURR], perm_thr);
        d_output_templaps[s] = d_output_overlaps[s];
    }
}

// =============================================================================
// Activate
// =============================================================================
void PatternClassifierDynamic::activate() {

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

    uint32_t total_count = 0;
    uint32_t max_count = 0;
    
    // loop through each stored label
    for (uint32_t l = 0; l < labels.size(); l++) {
        
        // zero label probability
        probs[l] = 0.0;

        // get count of active output statelets on the label state
        BitArray count_ba = l_states[l] & output[CURR];
        counts[l] = count_ba.count();

        // update total count
        total_count += counts[l];

        // store highest count value
        if (counts[l] > max_count) {
            max_count = counts[l];
        }
    }

    // update abnormality score based on highest count value
    // (represents how well the best matching label recognizes the input)
    pct_score = 1.0 - ((double)max_count / (double)num_as);

    // update probabilities
    if (total_count > 0) {
        for (uint32_t l = 0; l < labels.size(); l++) {
            probs[l] = (double)counts[l] / (double)total_count;
        }
    }
}

// =============================================================================
// Learn
// =============================================================================
void PatternClassifierDynamic::learn(const uint32_t label) {
    uint32_t idx = 0xFFFFFFFF;
    
    // check if input label exists in stored labels
    for (uint32_t l = 0; l < labels.size(); l++) {
        if (label == labels[l]) {
            idx = l;
        }
    }

    // if the label is unrecognized
    if (idx == 0xFFFFFFFF) {
        pct_score = 1.0;
        idx = (uint32_t)labels.size();

        // add new item to each respective array
        labels.push_back(label);
        counts.push_back(0);
        probs.push_back(0.0);
        l_states.emplace_back(BitArray(num_s));

        // assign labels to statelets
        double pct = (double)num_spl / (double)num_s;
        l_states[idx].random_fill(pct);
    }

    // get output activations
    std::vector<uint32_t> output_acts = output[CURR].get_acts();

    // for each active output statelet
    for (uint32_t k = 0; k < output_acts.size(); k++) {
        uint32_t s = output_acts[k];

        // shuffle learning mask
        lmask_ba.random_shuffle();

        // if active statelet has the correct label
        if (l_states[idx].get_bit(s) == 1) {
            
            // learn the statelet's coincidence set
            d_output[s].learn(input[CURR], lmask_ba, perm_inc, perm_dec);
        }

        // if statelet has the incorrect label
        else {

            // punish the statelet's coincidence set
            d_output[s].punish(input[CURR], lmask_ba, perm_inc);

            // learn a random statelet's coincidence set assigned to the correct label
            std::vector<uint32_t> acts = l_states[idx].get_acts();
            uint32_t rand = utils_rand_uint(0, (uint32_t)acts.size() - 1);
            uint32_t ss = acts[rand];
            d_output[ss].learn(input[CURR], lmask_ba, perm_inc, perm_dec);
        }
    }
}

/*
struct BitArray* PatternClassifierDynamic::decode() {
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