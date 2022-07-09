// =============================================================================
// scalar_transformer.cpp
// =============================================================================
#include "scalar_transformer.hpp"
#include <cassert>

using namespace BrainBlocks;

// =============================================================================
// # ScalarTransformer
//
// Converts a scalar value into a single binary representation.  It is useful
// for translating numerical data like "voltage", "temperature", etc. into a
// BitArray for processing higher level blocks.
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a ScalarTransformer.
// =============================================================================
ScalarTransformer::ScalarTransformer(
    const double min_val,  // minimum input value
    const double max_val,  // maximum input value
    const uint32_t num_s,  // number of statelets
    const uint32_t num_as, // number of active statelets
    const uint32_t num_t,  // number of BlockOutput time steps (optional)
    const uint32_t seed)     // seed for random number generator
: Block() {

    assert(min_val < max_val);
    assert(num_as < num_s);

    this->min_val = min_val;
    this->max_val = max_val;
    this->dif_val = max_val - min_val;
    this->num_s = num_s;
    this->num_as = num_as;
    this->dif_s = num_s - num_as;

    output.setup(num_t, num_s);
}

// =============================================================================
// # Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void ScalarTransformer::clear() {

    output.clear();
    value = 0.0;
    value_prev = 0.123456789;
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void ScalarTransformer::step() {

    output.step();
}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void ScalarTransformer::encode() {

    if (value != value_prev) {

        if (value < min_val) value = min_val;
        if (value > max_val) value = max_val;
        double percent = (value - min_val) / dif_val;
        uint32_t beg = (uint32_t)((double)dif_s * percent);

        output.state.clear_all();
        output.state.set_range(beg, num_as);
    }

    value_prev = value;
}

// =============================================================================
// # Decode
//
// Converts BlockOutput state(s) into BlockInput state(s).
// =============================================================================
void ScalarTransformer::decode() {

    // TODO: implement this
}


// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void ScalarTransformer::store() {

    output.store();
}
