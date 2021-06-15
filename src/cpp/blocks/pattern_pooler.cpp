// =============================================================================
// pattern_pooler.cpp
// =============================================================================
#include "pattern_pooler.hpp"
#include "../utils.hpp"

using namespace BrainBlocks;

// =============================================================================
// # PatternPooler
//
// Converts binary representations to sparse distributed binary representations.
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a PatternPooler.
// =============================================================================
PatternPooler::PatternPooler(
    const uint32_t num_s,   // number of statelets
    const uint32_t num_as,  // number of active statelets
    const uint8_t perm_thr, // receptor permanence threshold
    const uint8_t perm_inc, // receptor permanence increment
    const uint8_t perm_dec, // receptor permanence decrement
    const double pct_pool,  // percent pooled
    const double pct_conn,  // percent initially connected
    const double pct_learn, // percent learn
    const uint32_t num_t,   // number of BlockOutput time steps (optional)
    const bool always_update,  // whether to only update on input changes
    const uint32_t seed)    // seed for random number generator
: Block(seed) {

    assert(num_s > 0);
    assert(num_as > 0);

    this->num_s = num_s;
    this->num_as = num_as;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_pool = pct_pool;
    this->pct_conn = pct_conn;
    this->pct_learn = pct_learn;
    this->always_update = always_update;

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
void PatternPooler::init() {

    uint32_t num_i = input.state.num_bits();

    memory.init_pooled_conn(
        num_i, num_s, pct_pool, pct_conn, pct_learn, perm_thr, perm_inc,
        perm_dec, rng);

    init_flag = true;
}

// =============================================================================
// # Save
//
// Saves block memories.
// =============================================================================
bool PatternPooler::save(const char* file) {

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
bool PatternPooler::load(const char* file) {

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
void PatternPooler::clear() {

    input.clear();
    output.clear();
    memory.clear();
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void PatternPooler::step() {

    output.step();
}

// =============================================================================
// # Pull
//
// Updates BlockInput state(s) from child BlockOutput histories.
// =============================================================================
void PatternPooler::pull() {

    input.pull();
}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void PatternPooler::encode() {

    assert(init_flag);

    // If any BlockInput children have changed
    if (always_update || input.children_changed()) {

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
}

// =============================================================================
// # Learn
//
// Updates BlockMemories.
// =============================================================================
void PatternPooler::learn() {

    assert(init_flag);

    // If any BlockInput children have changed
    if (always_update || input.children_changed()) {

        std::vector<uint32_t> output_acts = output.state.get_acts();

        // Learn active statelets
        for (uint32_t k = 0; k < output_acts.size(); k++)
            memory.learn_conn(output_acts[k], input.state, rng);
    }
}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void PatternPooler::store() {

    output.store();
}
