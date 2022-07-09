// =============================================================================
// discrete_transformer.cpp
// =============================================================================
#include "discrete_transformer.hpp"
#include <cassert>

using namespace BrainBlocks;

// =============================================================================
// # DiscreteTransformer
//
// Converts a discrete numerical value into a single binary representation.
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a DiscreteTransformer.
// =============================================================================
DiscreteTransformer::DiscreteTransformer(
    const uint32_t num_v,  // number of discrete values
    const uint32_t num_s,  // number of statelets
    const uint32_t num_t,  // number of BlockOutput time steps (optional)
    const uint32_t seed)     // seed for random number generator
: Block(seed) {

    assert(num_v > 0);
    assert(num_s > 0);

    this->num_v = num_v;
    this->num_s = num_s;
    this->num_as = (uint32_t)((double)num_s / (double)num_v);
    this->dif_s = num_s - num_as;

    output.setup(num_t, num_s);
}

// =============================================================================
// # Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void DiscreteTransformer::clear() {

    output.clear();
    value = 0;
    value_prev = 0xFFFFFFFF;
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void DiscreteTransformer::step() {

    output.step();
}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void DiscreteTransformer::encode() {

    assert(value < num_v);

    if (value != value_prev) {

        double percent = (double)value / (double)(num_v - 1);
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
void DiscreteTransformer::decode() {

    // TODO: implement this
}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void DiscreteTransformer::store() {

    output.store();
}
