// =============================================================================
// discrete_transformer.cpp
// =============================================================================
#include "discrete_transformer.hpp"
#include <cassert>

using namespace BrainBlocks;

// =============================================================================
// # DiscreteTransformer
//
// Converts a discrete scalar value into a single binary representation.
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a DiscreteTransformer.
// =============================================================================
DiscreteTransformer::DiscreteTransformer(
    const uint32_t num_l,  // number of labels
    const uint32_t num_s,  // number of statelets
    const uint32_t num_t)  // number of BlockOutput time steps (optional)
: Block() {

    assert(num_l > 0);
    assert(num_s > 0);

    this->num_l = num_l;
    this->num_s = num_s;
    this->num_as = (uint32_t)((double)num_s / (double)num_l);
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

    assert(value < num_l);

    if (value != value_prev) {

        double percent = (double)value / (double)(num_l - 1);
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
