// =============================================================================
// persistence_transformer.cpp
// =============================================================================
#include "persistence_transformer.hpp"
#include <cassert>
#include <cmath>

using namespace BrainBlocks;

// =============================================================================
// # PersistenceTransformer
//
// TODO: add description
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a PersistenceTransformer.
// =============================================================================
PersistenceTransformer::PersistenceTransformer(
    const double min_val,
    const double max_val,
    const uint32_t num_s,
    const uint32_t num_as,
    const uint32_t max_step,
    const uint32_t num_t,
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
    this->max_step = max_step;
    this->counter = 0;
    this->pct_val_prev = 0.0;

    output.setup(num_t, num_s);
}

// =============================================================================
// # Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void PersistenceTransformer::clear() {

    output.clear();
    value = 0.0;
    counter = 0;
    pct_val_prev = 0.0;
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void PersistenceTransformer::step() {

    output.step();
}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void PersistenceTransformer::encode() {

    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;

    double pct_val = (value - min_val) / dif_val;
    double pct_delta = pct_val - pct_val_prev;
    bool reset_timer_flag = false;

    if (fabs(pct_delta) <= 0.1)
        counter += 1;
    else
        reset_timer_flag = true;

    if (counter >= max_step)
        counter = max_step;

    if (reset_timer_flag) {
        counter = 0;
        pct_val_prev = pct_val;
    }

    double pct_t = (double)counter / (double)max_step;
    uint32_t beg = (uint32_t)((double)dif_s * pct_t);

    output.state.clear_all();
    output.state.set_range(beg, num_as);
}

// =============================================================================
// # Decode
//
// Converts BlockOutput state(s) into BlockInput state(s).
// =============================================================================
void PersistenceTransformer::decode() {

    // TODO: implement this
}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void PersistenceTransformer::store() {

    output.store();
}
