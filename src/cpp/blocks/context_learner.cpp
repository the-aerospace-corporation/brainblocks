// =============================================================================
// context_learner.cpp
// =============================================================================
#include "context_learner.hpp"
#include "../utils.hpp"

using namespace BrainBlocks;

// =============================================================================
// # ContextLearner
//
// TODO: add description
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a ContextLearner.
// =============================================================================
ContextLearner::ContextLearner(
    const uint32_t num_c,    // number of column
    const uint32_t num_spc,  // number of statelets per column
    const uint32_t num_dps,  // number of dendrites per statelet
    const uint32_t num_rpd,  // number of receptors per dendrite
    const uint32_t d_thresh, // dendrite threshold
    const uint8_t perm_thr,  // receptor permanence threshold
    const uint8_t perm_inc,  // receptor permanence increment
    const uint8_t perm_dec,  // receptor permanence decrement
    const uint32_t num_t,    // number of BlockOutput time steps (optional)
    const uint32_t seed)     // seed for random number generator
: Block(seed) {

    assert(num_c > 0);
    assert(num_spc > 0);
    assert(num_dps > 0);
    assert(num_rpd > 0);
    assert(d_thresh < num_rpd);

    this->num_c = num_c;
    this->num_spc = num_spc;
    this->num_dps = num_dps;
    this->num_rpd = num_rpd;
    this->d_thresh = d_thresh;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;

    num_s = num_c * num_spc;
    num_d = num_s * num_dps;
    num_dpc = num_spc * num_dps;

    // Setup arrays
    next_sd.resize(num_s);
    d_used.resize(num_d);

    for (uint32_t s = 0; s < num_s; s++)
        next_sd[s] = 0;

    // Setup output
    output.setup(num_t, num_s);
}

// =============================================================================
// # Initialize
//
// Initializes BlockMemories based on BlockInput parameters.
// =============================================================================
void ContextLearner::init() {

    assert(input.state.num_bits() == num_c);

    uint32_t num_i = context.state.num_bits();
    double pct_learn = 1.0;

    memory.init(num_i, num_d, num_rpd, perm_thr, perm_inc, perm_dec, pct_learn);

    init_flag = true;
}

// =============================================================================
// # Save
//
// Saves block memories.
// =============================================================================
bool ContextLearner::save(const char* file) {

    FILE* fptr;

    // Check if file can be opened
    if ((fptr = std::fopen(file, "wb")) == NULL)
        return false;

    // Check if block has been initialized
    if (!init_flag)
        return false;

    // Save items
    memory.save(fptr);
    d_used.save(fptr);
    std::fwrite(next_sd.data(), sizeof(next_sd[0]), next_sd.size(), fptr);

    // Close file pointer
    std::fclose(fptr);

    return true;
}

// =============================================================================
// # Load
//
// Loads block memories.
// =============================================================================
bool ContextLearner::load(const char* file) {

    FILE* fptr;

    // Check if file can be opened
    if ((fptr = std::fopen(file, "rb")) == NULL)
        return false;

    // Check if block has been initialized
    if (!init_flag)
        init();

    // Load items
    memory.load(fptr);
    d_used.load(fptr);
    std::fread(next_sd.data(), sizeof(next_sd[0]), next_sd.size(), fptr);

    // Close file pointer
    std::fclose(fptr);

    return true;
}

// =============================================================================
// # Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void ContextLearner::clear() {

    input.clear();
    context.clear();
    output.clear();
    memory.clear();
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void ContextLearner::step() {

    output.step();
}

// =============================================================================
// # Pull
//
// Updates BlockInput state(s) from child BlockOutput histories.
// =============================================================================
void ContextLearner::pull() {

    input.pull();
    context.pull();
}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void ContextLearner::encode() {

    assert(init_flag);

    // If any BlockInput children have changed
    if (input.children_changed() || context.children_changed()) {

        // Get active columns
        input_acts = input.state.get_acts();

        // Clear data
        pct_anom = 0.0;
        output.state.clear_all();
        memory.state.clear_all();

        // For every active column
        for (uint32_t k = 0; k < input_acts.size(); k++) {
            uint32_t c = input_acts[k];
            surprise_flag = true;

            recognition(c);

            if (surprise_flag)
                surprise(c);
        }
    }
}

// =============================================================================
// # Learn
//
// Updates BlockMemories.
// =============================================================================
void ContextLearner::learn() {

    assert(init_flag);

    // If any BlockInput children have changed
    if (input.children_changed() || context.children_changed()) {

        // For every active column
        for (uint32_t k = 0; k < input_acts.size(); k++) {
            uint32_t c = input_acts[k];
            uint32_t d_beg = c * num_dpc;
            uint32_t d_end = d_beg + num_dpc;

            // For every dendrite on the column
            for (uint32_t d = d_beg; d < d_end; d++) {

                // Learn and move the dendrite if it is active
                if (memory.state.get_bit(d)) {
                    memory.learn_move(d, context.state, rng);
                    d_used.set_bit(d);
                }
            }
        }
    }
}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void ContextLearner::store() {

    output.store();
}

// =============================================================================
// # Recognition
//
// TODO: add description
// =============================================================================
void ContextLearner::recognition(const uint32_t c) {

    uint32_t d_beg = c * num_dpc;
    uint32_t d_end = d_beg + num_dpc;

    // For every dendrite on the column
    for (uint32_t d = d_beg; d < d_end; d++) {

        // If dendrite is used then overlap
        if (d_used.get_bit(d)) {

            // Overlap dendrite with context
            uint32_t overlap = memory.overlap(d, context.state);

            // If dendrite overlap is above the threshold
            if (overlap >= d_thresh) {
                uint32_t s = d / num_dps;
                memory.state.set_bit(d); // activate the dendrite
                output.state.set_bit(s); // activate the dendrite's statelet
                surprise_flag = false;
            }
        }
    }
}

// =============================================================================
// # Suprise
//
// TODO: add description
// =============================================================================
void ContextLearner::surprise(const uint32_t c) {

    // Update abnormality score
    pct_anom += (1.0 / input_acts.size());

    // Get statelet index information
    uint32_t s_beg = c * num_spc;
    uint32_t s_end = s_beg + num_spc - 1;
    uint32_t s_rand = utils_rand_uint(s_beg, s_end, rng);

    // Activate random statelet
    output.state.set_bit(s_rand);

    // Activate random statelet's next available dendritet
    set_next_available_dendrite(s_rand);
}

// =============================================================================
// # Set Next Available Dendrite
//
// TODO: add description
// =============================================================================
void ContextLearner::set_next_available_dendrite(const uint32_t s) {

    // Get dendrite index information
    uint32_t d_beg = s * num_dps;
    uint32_t d_next = d_beg + next_sd[s];

    // Activate random statelet's next available dendrite
    memory.state.set_bit(d_next);

    // Update random statelet's next available dendrite
    if(next_sd[s] < num_dps - 1)
        next_sd[s]++;
}
