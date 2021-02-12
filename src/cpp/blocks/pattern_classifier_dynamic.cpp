// =============================================================================
// pattern_classifier_dynamic.cpp
// =============================================================================
#include "pattern_classifier_dynamic.hpp"
#include "../utils.hpp"

using namespace BrainBlocks;

// =============================================================================
// # PatternClassifierDynamic
//
// TODO: add description
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a PatternClassifierDynamic.
// =============================================================================
PatternClassifierDynamic::PatternClassifierDynamic(
    const uint32_t num_s,   // number of statelets
    const uint32_t num_as,  // number of active statelets
    const uint32_t num_spl, // number of statelets per label
    const uint8_t perm_thr, // receptor permanence threshold
    const uint8_t perm_inc, // receptor permanence increment
    const uint8_t perm_dec, // receptor permanence decrement
    const double pct_pool,  // percent pooled
    const double pct_conn,  // percent initially connected
    const double pct_learn, // percent learn
    const uint32_t num_t,   // number of BlockOutput time steps (optional)
    const uint32_t seed)    // seed for random number generator
: Block(seed) {

    assert(num_s > 0);
    assert(num_as > 0);
    assert(num_spl > 0);

    this->num_s = num_s;
    this->num_as = num_as;
    this->num_spl = num_spl;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_pool = pct_pool;
    this->pct_conn = pct_conn;
    this->pct_learn = pct_learn;

    overlaps.resize(num_s);
    templaps.resize(num_s);

    // Setup output
    output.setup(num_t, num_s);
}

// =============================================================================
// # Initialize
//
// Initializes BlockMemories based on BlockInput parameters.
// =============================================================================
void PatternClassifierDynamic::init() {

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
bool PatternClassifierDynamic::save(const char* file) {

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
bool PatternClassifierDynamic::load(const char* file) {

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
void PatternClassifierDynamic::clear() {

    input.clear();
    output.clear();
    memory.clear();
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void PatternClassifierDynamic::step() {

    output.step();
}

// =============================================================================
// # Pull
//
// Updates BlockInput state(s) from child BlockOutput histories.
// =============================================================================
void PatternClassifierDynamic::pull() {

    input.pull();
}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void PatternClassifierDynamic::encode() {

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
void PatternClassifierDynamic::learn() {

    assert(init_flag);

    uint32_t idx = 0xFFFFFFFF;

    // Check if label exists in stored labels
    for (uint32_t l = 0; l < labels.size(); l++)
        if (label == labels[l])
            idx = l;

    // If label is unrecognized
    if (idx == 0xFFFFFFFF) {
        pct_anom = 1.0;
        idx = (uint32_t)labels.size();

        // Add new item to arrays
        labels.push_back(label);
        counts.push_back(0);
        l_states.emplace_back(BitArray(num_s));

        // Randomly assign labels to statelets
        l_states[idx].random_set_num(rng, num_spl);
    }

    std::vector<uint32_t> output_acts = output.state.get_acts();

    // Loop through active statelets
    for (uint32_t k = 0; k < output_acts.size(); k++) {
        uint32_t s = output_acts[k];

        // Learn if active statelet is the correct label
        if (l_states[idx].get_bit(s))
            memory.learn_conn(s, input.state, rng);

        // If active statelet is not the correct label
        else {

            // Punish
            memory.punish_conn(s, input.state, rng);

            // Learn a random statelet from l_state
            uint32_t rand = utils_rand_uint(0, num_s - 1, rng);
            uint32_t s_rand = 0xFFFFFFFF;
            l_states[idx].find_next_set_bit(rand, &s_rand);
            memory.learn_conn(s_rand, input.state, rng);
        }
    }
}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void PatternClassifierDynamic::store() {

    output.store();
}

// =============================================================================
// Get Probabilities
//
// Returns array of probability scores for each stored label.
// =============================================================================
std::vector<double> PatternClassifierDynamic::get_probabilities() {

    uint32_t total_count = 0;
    uint32_t max_count = 0;
    uint32_t num_l = (uint32_t)labels.size();
    std::vector<double> probs(num_l);

    // Loop through stored labels
    for (uint32_t l = 0; l < num_l; l++) {

        probs[l] = 0.0;

        // Get count of active output statelets of the label state
        counts[l] = l_states[l].num_similar(output.state);

        // Update total count
        total_count += counts[l];

        // Store highest count
        if (counts[l] > max_count)
            max_count = counts[l];
    }

    // Update abnormality score based on highest count value
    pct_anom = 1.0 - ((double)max_count / (double)num_as);

    // Update probabilities
    if (total_count > 0)
        for (uint32_t l = 0; l < num_l; l++)
            //probs[l] = (double)counts[l] / (double)total_count;
            probs[l] = (double)counts[l] / (double)num_as;

    return probs;
}
