#include "context_learner.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdio>

// =============================================================================
// Constructor
// =============================================================================
ContextLearner::ContextLearner(
        const uint32_t num_spc,
        const uint32_t num_dps,
        const uint32_t num_rpd,
        const uint32_t d_thresh,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec) {

    // error check
    if (num_spc == 0) {
        std::cout << "Error in ContextLearner::ContextLearner(): num_spc == 0" << std::endl;
        exit(1);
    }

    if (num_dps == 0) {
        std::cout << "Error in ContextLearner::ContextLearner(): num_dps == 0" << std::endl;
        exit(1);
    }

    if (num_rpd == 0) {
        std::cout << "Error in ContextLearner::ContextLearner(): num_rpd == 0" << std::endl;
        exit(1);
    }

    if (d_thresh > num_rpd) {
        std::cout << "Error in ContextLearner::ContextLearner(): d_thresh > num_rpd" << std::endl;
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        std::cout << "Error in ContextLearner::ContextLearner(): perm_thr > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        std::cout << "Error in ContextLearner::ContextLearner(): perm_inc > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        std::cout << "Error in ContextLearner::ContextLearner(): perm_dec > PERM_MAX" << std::endl;
        exit(1);
    }

    // setup variables
    this->num_c = 0;
    this->num_s = 0;
    this->num_d = 0;
    this->num_spc = num_spc;
    this->num_dps = num_dps;
    this->num_dpc = num_spc * num_dps;
    this->num_rpd = num_rpd;
    this->d_thresh = d_thresh;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_score = 0.0;
    this->init_flag = false;

    // setup input page
    input.set_num_bitarrays(2);
    input.set_num_bits(0);

    // setup context page
    context.set_num_bitarrays(2);
    context.set_num_bits(0);
}

// =============================================================================
// Initialize
// =============================================================================
void ContextLearner::initialize() {

    // initialize input page
    input.initialize();
    context.initialize();

    // update variables
    num_c = input.get_num_bits();
    num_s = num_c * num_spc;
    num_d = num_s * num_dps;

    // setup and initialize output
    output.set_num_bitarrays(2);
    output.set_num_bits(num_s);
    output.initialize();

    // setup array of next coincidence sets per statelet
    s_next_d.resize(num_s);

    // setup output coincidence detector
    d_output.resize(num_d);
    d_output_overlaps.resize(num_d);
    d_output_states.resize(num_d);

    for (uint32_t d = 0; d < num_d; d++) {
        d_output[d].resize(num_rpd);
    }

    // setup learning mask bitarray
    lmask_ba.resize(num_rpd);
    lmask_ba.fill_bits();

    // set init_flag to true
    init_flag = true;
}

// =============================================================================
// Save
// =============================================================================
void ContextLearner::save(const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"wb")) == NULL) {
       std::cout << "Error in ContextLearner::save(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        std::cout << "Error in ContextLearner::save(): block not initialized" << std::endl;
    }

    CoincidenceSet* cs;
    
    // save output coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < num_d; d++) {
        cs = &d_output[d];
        fwrite(cs->get_addrs().data(), cs->get_num_r() * sizeof(cs->get_addrs()[0]), 1, fptr);
        fwrite(cs->get_perms().data(), cs->get_num_r() * sizeof(cs->get_perms()[0]), 1, fptr);        
    }

    // save next available coincidence detector on each statelet
    fwrite(s_next_d.data(), num_s * sizeof(s_next_d[0]), 1, fptr);

    fclose(fptr);
}

// =============================================================================
// Load
// =============================================================================
void ContextLearner::load(const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"rb")) == NULL) {
       std::cout << "Error in ContextLearner::load(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        initialize();
    }

    CoincidenceSet* cs;

    // load output coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < num_d; d++) {
        cs = &d_output[d];
        fread(cs->get_addrs().data(), cs->get_num_r() * sizeof(cs->get_addrs()[0]), 1, fptr);
        fread(cs->get_perms().data(), cs->get_num_r() * sizeof(cs->get_perms()[0]), 1, fptr);        
    }

    // load next available coincidence detector on each statelet
    fread(s_next_d.data(), num_s * sizeof(s_next_d[0]), 1, fptr);

    fclose(fptr); 
}

// =============================================================================
// Clear
// =============================================================================
void ContextLearner::clear_states() {
    input[CURR].clear_bits();
    input[PREV].clear_bits();
    context[CURR].clear_bits();
    context[PREV].clear_bits();
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}

// =============================================================================
// Compute
// =============================================================================
void ContextLearner::compute(const bool learn_flag) {
    if (init_flag == false) {
        initialize();
    }

    input.step();
    context.step();
    output.step();
    input.fetch();
    context.fetch();

    if (input.has_changed() || context.has_changed()) {
        overlap();
        activate(learn_flag);

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
void ContextLearner::overlap() {

    // get input activations
    std::vector<uint32_t> input_acts = input[CURR].get_acts();

    // for every active column
    for (uint32_t k = 0; k < input_acts.size(); k++) {
        uint32_t c = input_acts[k];

        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < num_dpc; cd++) {
            uint32_t d = cd + (c * num_dpc);

            // overlap output coincidence detectors with current context state
            d_output_overlaps[d] = d_output[d].overlap(context[CURR], perm_thr);
        }
    }
}

// =============================================================================
// Activate
// =============================================================================
void ContextLearner::activate(const uint32_t learn_flag) {
    pct_score = 0.0;

    // get input activations
    std::vector<uint32_t> input_acts = input[CURR].get_acts();

    // zero all output coincidence detector states
    d_output_states.clear_bits();

    // for every active column
    for (uint32_t k = 0; k < input_acts.size(); k++) {
        uint32_t c = input_acts[k];
        bool output_surprise_flag = true;

        // ====================
        // Recognition
        // ====================

        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < num_dpc; cd++) {

            // get global index of the coincidence detector
            uint32_t d = cd + (c * num_dpc); // global index of coincidence detector

            // if output coincidence detector overlap is above the threshold
            if (d_output_overlaps[d] >= d_thresh) {
                uint32_t s = d / num_dps;      // get global index of statelet
                d_output_states.set_bit(d, 1); // activate output coincidence detector
                output[CURR].set_bit(s, 1);    // activate output statelet
                output_surprise_flag = false;
            }
        }

        // ====================
        // Surprise
        // ====================
        if (output_surprise_flag) {
            pct_score++;

            uint32_t s_beg = c * num_spc;                    // global index of first statelet 
            uint32_t s_end = s_beg + num_spc - 1;            // global index of final statelet 
            uint32_t s_rand = utils_rand_uint(s_beg, s_end); // global index of random statelet

            // activate output statelet
            output[CURR].set_bit(s_rand, 1);

            if (learn_flag) {
                uint32_t d_beg = s_rand * num_dps;          // global index of first coincidence detector on random statelet
                uint32_t d_next = d_beg + s_next_d[s_rand]; // global index of next available coincidence detector on random statelet

                // activate next available hidden and output coincidence detector
                d_output_states.set_bit(d_next, 1);

                // if next available coincidence detector is less than the number of coincidence detectors per statelet
                if (s_next_d[s_rand] < num_dps) {
                    
                    // update next available coincidence detector on the random statelet
                    s_next_d[s_rand]++;
                }
            }
        }
    }

    pct_score = (double)pct_score / (double)input_acts.size();
}

// =============================================================================
// Learn
// =============================================================================
void ContextLearner::learn() {

    // get input activations
    std::vector<uint32_t> input_acts = input[CURR].get_acts();

    // for every active column
    for (uint32_t k = 0; k < input_acts.size(); k++) {
        uint32_t c = input_acts[k];

        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < num_dpc; cd++) {
            
            // get global index of the coincidence detector
            uint32_t d = cd + (c * num_dpc);

            // if output coincidence set is active then learn
            if (d_output_states.get_bit(d) == 1) {
                d_output[d].learn_move(context[CURR], lmask_ba, perm_inc, perm_dec, perm_thr);
            }
        }
    }
}