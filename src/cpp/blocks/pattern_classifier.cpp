// =============================================================================
// pattern_classifier.cpp
// =============================================================================
#include "pattern_classifier.hpp"
#include "../utils.hpp"
#include <cstring> // for memset
#include <iostream>

using namespace BrainBlocks;

// =============================================================================
// # PatternClassifier
//
// TODO: add description
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a PatternClassifier.
// =============================================================================
PatternClassifier::PatternClassifier(
    const uint32_t num_l,   // number of labels
    const uint32_t num_s,   // number of statelets
    const uint32_t num_as,  // number of active statelets
    const uint8_t perm_thr, // receptor permanence threshold
    const uint8_t perm_inc, // receptor permanence increment
    const uint8_t perm_dec, // receptor permanence decrement
    const double pct_pool,  // percent pooled
    const double pct_conn,  // percent initially connected
    const double pct_learn, // percent learn
    const uint32_t num_t,   // number of BlockOutput time steps (optional)
    const uint32_t seed)    // seed for random number generator
: Block(seed) {

    assert(num_l > 0);
    assert(num_s > 0);
    assert(num_as > 0);

    this->num_l = num_l;
    this->num_s = num_s;
    this->num_as = num_as;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_pool = pct_pool;
    this->pct_conn = pct_conn;
    this->pct_learn = pct_learn;

    num_spl = (uint32_t)((double)num_s / (double)num_l);

    overlaps.resize(num_s);
    templaps.resize(num_s);
    s_labels.resize(num_s);

    // Setup statelet labels
    for (uint32_t s = 0; s < num_s; s++) {
        uint32_t label = (uint32_t)((double)s / (double)num_spl);

        // Ensure statelet label does not exceed the number of labels
        if (label >= num_l)
            label = 0;

        s_labels[s] = label;
    }

    // Setup output
    output.setup(num_t, num_s);
}

// =============================================================================
// # Initialize
//
// Initializes BlockMemories based on BlockInput parameters.
// =============================================================================
void PatternClassifier::init() {

    uint32_t num_i = input.state.num_bits();

    memory.init_pooled_conn(
        num_i, num_s, pct_pool, pct_conn, pct_learn,
        perm_thr, perm_inc, perm_dec, rng);

    init_flag = true;
}

// =============================================================================
// # Save
//
// Saves block memories.
// =============================================================================
bool PatternClassifier::save(const char* file) {

    FILE* fptr;

    // Check if file can be opened
    if ((fptr = std::fopen(file, "wb")) == NULL)
        return false;

    // Check if block has been initialized
    if (!init_flag)
        return false;

    // Save items
    memory.save(fptr);

    // Close file pointer
    std::fclose(fptr);

    return true;
}

// =============================================================================
// # Load
//
// Loads block memories.
// =============================================================================
bool PatternClassifier::load(const char* file) {

    FILE* fptr;

    // Check if file can be opened
    if ((fptr = std::fopen(file, "rb")) == NULL)
        return false;

    // Check if block has been initialized
    if (!init_flag)
        init();

    // Load items
    memory.load(fptr);

    // Close file pointer
    std::fclose(fptr);

    return true;
}

// =============================================================================
// # Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void PatternClassifier::clear() {

    input.clear();
    output.clear();
    memory.clear();
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void PatternClassifier::step() {

    output.step();
}

// =============================================================================
// # Pull
//
// Updates BlockInput state(s) from child BlockOutput histories.
// =============================================================================
void PatternClassifier::pull() {

    input.pull();
}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void PatternClassifier::encode() {

    assert(init_flag);

    // Clear data
    output.state.clear_all();

    // Overlap each statelet
    for (uint32_t s = 0; s < num_s; s++) {
        overlaps[s] = memory.overlap_conn(s, input.state);
        templaps[s] = overlaps[s];
    }

    // Activate statelets with k-highest overlap
    for (uint32_t k = 0; k < num_as; k++) {
        //uint32_t beg_idx = utils_rand_uint(0, num_s - 1);
        uint32_t beg_idx = 0;
        uint32_t max_val = 0;
        uint32_t max_idx = beg_idx;

        // Loop through statelets
        for (uint32_t i = 0; i < num_s; i++) {

            // Account for random start location and wrap if necessary
            //uint32_t j = i + beg_idx;
            //uint32_t s = (j < num_s) ? j : (j - num_s);
            uint32_t s = i;

            // Find statelet with highest overlap
            if (templaps[s] > max_val) {
                max_val = templaps[s];
                max_idx = s;
            }
        }

        // Activate statelet with highest overlap
        output.state.set_bit(max_idx);
        templaps[max_idx] = 0;
    }
}

// =============================================================================
// # Learn
//
// Updates BlockMemories.
// =============================================================================
void PatternClassifier::learn() {

    assert(init_flag);
    assert(label < num_l);

    std::vector<uint32_t> output_acts = output.state.get_acts();

    // Loop through active statelets
    for (uint32_t k = 0; k < output_acts.size(); k++) {
        uint32_t s = output_acts[k];

        // Learn if label matches statelet label
        if (label == s_labels[s])
            memory.learn_conn(s, input.state, rng);

        // If label does not match statelet label
        else {

            // Punish
            memory.punish_conn(s, input.state, rng);

            // TODO: works but need to verify on moons & blobs sklearn datasets
            // Learn a random statelet from l_states
            //uint32_t beg = label * num_spl;
            //uint32_t end = beg + num_spl - 1;
            //uint32_t s_rand = utils_rand_uint(beg, end, rng);
            //memory.learn_conn(s_rand, input.state, rng);
        }
    }
}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void PatternClassifier::store() {

    output.store();
}

// =============================================================================
// # Get Labels
//
// Returns array of stored labels.
// =============================================================================
std::vector<uint32_t> PatternClassifier::get_labels() {

    std::vector<uint32_t> labels(num_l);

    for (uint32_t l = 0; l < num_l; l++)
        labels[l] = l;

    return labels;
}

// =============================================================================
// # Get Probabilities
//
// Returns array of probability scores for each stored label.
// =============================================================================
std::vector<double> PatternClassifier::get_probabilities() {

    double prob_inc = 1.0 / (double)num_as;
    std::vector<uint32_t> output_acts = output.state.get_acts();
    std::vector<double> probs(num_l);

    // Zero probabilities
    memset(probs.data(), 0, probs.size() * sizeof(probs[0]));

    // Increment probabilities based on output activations
    for (uint32_t i = 0; i < output_acts.size(); i++)
        probs[s_labels[output_acts[i]]] += prob_inc;

    return probs;
}
