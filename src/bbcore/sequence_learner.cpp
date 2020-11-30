#include "sequence_learner.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdio>

// =============================================================================
// Constructor
// =============================================================================
SequenceLearner::SequenceLearner(
        const uint32_t num_spc,
        const uint32_t num_dps,
        const uint32_t num_rpd,
        const uint32_t d_thresh,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec) {

    // error check
    if (num_spc == 0) {
        std::cout << "Error in SequenceLearner::SequenceLearner(): num_spc == 0" << std::endl;
        exit(1);
    }

    if (num_dps == 0) {
        std::cout << "Error in SequenceLearner::SequenceLearner(): num_dps == 0" << std::endl;
        exit(1);
    }

    if (num_rpd == 0) {
        std::cout << "Error in SequenceLearner::SequenceLearner(): num_rpd == 0" << std::endl;
        exit(1);
    }

    if (d_thresh > num_rpd) {
        std::cout << "Error in SequenceLearner::SequenceLearner(): d_thresh > num_rpd" << std::endl;
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        std::cout << "Error in SequenceLearner::SequenceLearner(): perm_thr > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_inc > PERM_MAX) {
        std::cout << "Error in SequenceLearner::SequenceLearner(): perm_inc > PERM_MAX" << std::endl;
        exit(1);
    }

    if (perm_dec > PERM_MAX) {
        std::cout << "Error in SequenceLearner::SequenceLearner(): perm_dec > PERM_MAX" << std::endl;
        exit(1);
    }

    // setup variables
    this->num_c = 0;
    this->num_spc = num_spc;
    this->num_dps = num_dps;
    this->num_dpc = num_spc * num_dps;
    this->num_s = 0;
    this->num_d = 0;
    this->num_rpd = num_rpd;
    this->d_thresh = d_thresh;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->count_hs = 0;
    this->count_hd = 0;
    this->pct_score = 0.0;
    this->init_flag = false;

    // setup input page (output and hidden constructed in initialize())
    input.set_num_bitarrays(2);
    input.set_num_bits(0);
}

// =============================================================================
// Initialize
// =============================================================================
void SequenceLearner::initialize() {

    // initialize input page
    input.initialize();

    // update variables
    num_c = input.get_num_bits();
    num_s = num_c * num_spc;
    num_d = num_s * num_dps;

    // setup and initialize output and hidden pages
    hidden.set_num_bitarrays(2);
    output.set_num_bitarrays(2);
    hidden.set_num_bits(num_s);
    output.set_num_bits(num_s);
    hidden.initialize();
    output.initialize();

    // setup array of next coincidence sets per statelet
    s_next_d.resize(num_s);

    // setup hidden and output coincidence detectors
    d_hidden.resize(num_d);
    d_output.resize(num_d);
    d_hidden_overlaps.resize(num_d);
    d_output_overlaps.resize(num_d);
    d_hidden_states.resize(num_d);
    d_output_states.resize(num_d);

    for (uint32_t d = 0; d < num_d; d++) {
        d_hidden[d].resize(num_rpd);
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
void SequenceLearner::save(const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"wb")) == NULL) {
       std::cout << "Error in SequenceLearner::save(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        std::cout << "Error in SequenceLearner::save(): block not initialized" << std::endl;
    }

    CoincidenceSet* cs;

    // save hidden coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < num_d; d++) {
        cs = &d_hidden[d];
        fwrite(cs->get_addrs().data(), cs->get_num_r() * sizeof(cs->get_addrs()[0]), 1, fptr);
        fwrite(cs->get_perms().data(), cs->get_num_r() * sizeof(cs->get_perms()[0]), 1, fptr);
    }
    
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
void SequenceLearner::load(const char* file) {
    FILE *fptr;

    // check if file can be opened
    if ((fptr = fopen(file,"rb")) == NULL) {
       std::cout << "Error in SequenceLearner::load(): cannot open file" << std::endl;
       exit(1);
    }

    // check if block has been initialized
    if (init_flag == false) {
        initialize();
    }

    CoincidenceSet* cs;

    // load hidden coincidence detector receptor addresses and permanences
    for (uint32_t d = 0; d < num_d; d++) {
        cs = &d_hidden[d];
        fread(cs->get_addrs().data(), cs->get_num_r() * sizeof(cs->get_addrs()[0]), 1, fptr);
        fread(cs->get_perms().data(), cs->get_num_r() * sizeof(cs->get_perms()[0]), 1, fptr);
    }

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
void SequenceLearner::clear_states() {
    input[CURR].clear_bits();
    input[PREV].clear_bits();
    hidden[CURR].clear_bits();
    hidden[PREV].clear_bits();
    output[CURR].clear_bits();
    output[PREV].clear_bits();
}

// =============================================================================
// Compute
// =============================================================================
void SequenceLearner::compute(const bool learn_flag) {
    if (init_flag == false) {
        initialize();
    }

    input.step();
    hidden.step();
    output.step();
    input.fetch();

    if (input.has_changed() || hidden.has_changed()) {
        overlap();
        activate(learn_flag);

        if (learn_flag) {
            learn();
        }

        hidden.compute_changed();
        output.compute_changed(); // TODO: might not be needed since hidden states are the primary driver
    }
    else {
        hidden.copy_previous_to_current();
        output.copy_previous_to_current();
        hidden.set_changed_flag(false);
        output.set_changed_flag(false);
    }
}

/*
// =============================================================================
// Get Historical Statelets
// =============================================================================
struct BitArray* sequence_learner_get_historical_statelets(struct SequenceLearner* sl) {
    struct BitArray* hist_ba = (BitArray*)malloc(sizeof(*hist_ba)); // TODO: how do I free this?
    bitarray_construct(hist_ba, sl->num_s);
    
    for (uint32_t s = 0; s < sl->num_s; s++) {
        if (sl->s_next_d[s] > 0) {
            bitarray_set_bit(hist_ba, s);
        }
    }
    
    return hist_ba;
}
*/

// =============================================================================
// Overlap
// =============================================================================
void SequenceLearner::overlap() {

    // get input activations
    std::vector<uint32_t> input_acts = input[CURR].get_acts();

    // for every active column
    for (uint32_t k = 0; k < input_acts.size(); k++) {
        uint32_t c = input_acts[k];

        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < num_dpc; cd++) {
            uint32_t d = cd + (c * num_dpc);

            // overlap hidden and output coincidence detectors
            // with previous hidden state
            d_hidden_overlaps[d] = d_hidden[d].overlap(hidden[PREV], perm_thr);
            d_output_overlaps[d] = d_output[d].overlap(hidden[PREV], perm_thr);
        }
    }
}

// =============================================================================
// Activate
// =============================================================================
void SequenceLearner::activate(const uint32_t learn_flag) {
    pct_score = 0.0;

    // get input activations
    std::vector<uint32_t> input_acts = input[CURR].get_acts();

    // zero all output coincidence detector states
    d_hidden_states.clear_bits();
    d_output_states.clear_bits();

    // for every active column
    for (uint32_t k = 0; k < input_acts.size(); k++) {
        uint32_t c = input_acts[k];
        bool hidden_surprise_flag = true;
        bool output_surprise_flag = true;

        // ====================
        // Recognition
        // ====================
        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < num_dpc; cd++) {

            // get global index of the coincidence detector
            uint32_t d = cd + (c * num_dpc); // global index of coincidence detector

            // if hidden coincidence detector overlap is above the threshold
            if (d_hidden_overlaps[d] >= d_thresh) {
                uint32_t s = d / num_dps;      // get global index of statelet
                d_hidden_states.set_bit(d, 1); // activate hidden coincidence detector
                hidden[CURR].set_bit(s, 1);    // activate hidden statelet
                hidden_surprise_flag = false;
            }

            // if output coincidence detector overlap is above the threshold
            if (d_output_overlaps[d] >= d_thresh) {
                uint32_t s = d / num_dps;      // get global index of statelet
                d_output_states.set_bit(d, 1); // activate output coincidence detector
                output[CURR].set_bit(s, 1);    // activate output statelet
                output_surprise_flag = false;
            }
        }

        /*
        // handles instances where hidden states closed loops with historical statelets
        // but output statelets never empirically learned the transition
        if (!hidden_surprise_flag && output_surprise_flag) {
            pct_score++;
            
            uint32_t s_beg = c * num_spc;                    // global index of first statelet 
            uint32_t s_end = s_beg + num_spc - 1;            // global index of final statelet 
            uint32_t s_rand = utils_rand_uint(s_beg, s_end); // global index of random statelet

            // activate random output statelet
            output[CURR].set_bit(s_rand, 1);

            // activate next available output coincidence detector
            if (learn_flag) {
                uint32_t d_beg = s_rand * num_dps;          // global index of first coincidence detector on random statelet
                uint32_t d_next = d_beg + s_next_d[s_rand]; // global index of next available coincidence detector on random statelet

                // activate next available hidden and output coincidence detector
                d_output_states.set_bit(d_next, 1);

                // if next available coincidence detector is less than the number of coincidence detectors per statelet
                if (s_next_d[s_rand] < num_dps) {

                    // update historical coincidence detector and statelet counters
                    // remember: both hidden and output were activated in this case
                    count_hd += 1;
                    if (s_next_d[s_rand] == 0) {
                        count_hs += 1;
                    }
                    
                    // update next available coincidence detector on the random statelet
                    s_next_d[s_rand]++;
                }
            }
        }
        */

        // ====================
        // Surprise
        // ====================
        if (hidden_surprise_flag) {
            pct_score++;

            uint32_t s_beg = c * num_spc;                    // global index of first statelet 
            uint32_t s_end = s_beg + num_spc - 1;            // global index of final statelet 
            uint32_t s_rand = utils_rand_uint(s_beg, s_end); // global index of random statelet

            // activate random hidden and output statelets
            hidden[CURR].set_bit(s_rand, 1);
            output[CURR].set_bit(s_rand, 1);

            // activate next available hidden and output coincidence detectors
            if (learn_flag) {
                uint32_t d_beg = s_rand * num_dps;          // global index of first coincidence detector on random statelet
                uint32_t d_next = d_beg + s_next_d[s_rand]; // global index of next available coincidence detector on random statelet

                // activate next available hidden and output coincidence detector
                d_hidden_states.set_bit(d_next, 1);
                d_output_states.set_bit(d_next, 1);

                // if next available coincidence detector is less than the number of coincidence detectors per statelet
                if (s_next_d[s_rand] < num_dps - 1) {

                    // update historical coincidence detector and statelet counters
                    // remember: both hidden and output were activated in this case
                    count_hd += 2; 
                    if (s_next_d[s_rand] == 0) {
                        count_hs += 2;
                    }
                    
                    // update next available coincidence detector on the random statelet
                    s_next_d[s_rand]++;
                }
            }

            // activate all hidden historical statelets
            // for each global index of statelets on the column
            for (uint32_t s = s_beg; s <= s_end; s++) {

                // if the statelet is not the random statelet
                // and the statelet has at lease one coincidence detector
                if (s != s_rand && s_next_d[s] > 0) {

                    // activate the hidden statlet
                    hidden[CURR].set_bit(s, 1);

                    // activate next available hidden coincidence detector
                    if (learn_flag) {
                        uint32_t d_beg = s * num_dps;
                        uint32_t d_next = d_beg + s_next_d[s];

                        // activate next available hidden coincidence detector
                        d_hidden_states.set_bit(d_next, 1);

                        if (s_next_d[s] < num_dps-1) {
                            count_hd++;
                            if (s_next_d[s] == 0) {
                                count_hs++;
                            }

                            s_next_d[s]++;
                        }
                    }
                }
            }
        }
    }

    pct_score = (double)pct_score / (double)input_acts.size();
}

// =============================================================================
// Learn
// =============================================================================
void SequenceLearner::learn() {

    // get input activations
    std::vector<uint32_t> input_acts = input[CURR].get_acts();

    // for every active column
    for (uint32_t k = 0; k < input_acts.size(); k++) {
        uint32_t c = input_acts[k];

        // for every coincidence detector on the active column
        for (uint32_t cd = 0; cd < num_dpc; cd++) {
            
            // get global index of the coincidence detector
            uint32_t d = cd + (c * num_dpc);

            // if hidden coincidence set is active then learn
            if (d_hidden_states.get_bit(d) == 1) {
                d_hidden[d].learn_move(hidden[PREV], lmask_ba, perm_inc, perm_dec, perm_thr);
            }

            // if output coincidence set is active then learn
            if (d_output_states.get_bit(d) == 1) {
                d_output[d].learn_move(output[PREV], lmask_ba, perm_inc, perm_dec, perm_thr);
            }            
        }
    }
}