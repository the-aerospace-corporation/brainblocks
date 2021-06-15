// =============================================================================
// block_output.cpp
// =============================================================================
#include "block_output.hpp"
#include <cassert>

using namespace BrainBlocks;

uint32_t BlockOutput::next_id = 0;

// =============================================================================
// Constructor
//
// Constructs a BlockOutput.
// =============================================================================
BlockOutput::BlockOutput() {

    this->id = next_id++;
}

// =============================================================================
// # Setup
//
// Sets up the BlockOutput based on:
//
// - num_t: the number of time steps (aka history)
// - num_b: number of bits per BitArray
//
// ## Example
//
// BlockOutput output;
//
// output.setup(3, 32);
//
// Resulting BlockOutput
// ---------------------
// state:                               changed_flag:
// {00000000000000000000000000000000}   true
//
// history:                             changes:
// {00000000000000000000000000000000}   true
// {00000000000000000000000000000000}   true
// {00000000000000000000000000000000}   true
// =============================================================================
void BlockOutput::setup(const uint32_t num_t, const uint32_t num_b) {

    assert(num_t >= 2);
    assert(num_b > 0);

    // If the number of bits is not divisible 32 then round up.
    // TODO: fix bitarray_copy() to do bit-indexing instead of word-indexing
    uint32_t num_bits = num_b;

    if (num_b % 32 != 0)
        num_bits = (num_b + 31) & -32;

    // resize vectors
    state.resize(num_bits);
    history.resize(num_t);
    changes.resize(num_t);

    for (uint32_t i = 0; i < history.size(); i++) {
        history[i].resize(num_b);
        changes[i] = true;
    }
}

// =============================================================================
// # Clear
//
// Set all bits to 0 in the state BitArray and each BitArray in history vector.
// =============================================================================
void BlockOutput::clear() {

    state.clear_all();
    changed_flag = true;

    for (uint32_t i = 0; i < history.size(); i++) {
        history[i].clear_all();
        changes[i] = true;
    }
}

// =============================================================================
// # Step
//
// Updates current index variable.
//
// ## Example
//
// | idx | history    | t |
// |-----|------------|---|
// |  0  | {11000000} | 0 | <-- curr_idx
// |  1  | {00110000} | 1 |
// |  2  | {00001100} | 2 |
//
// out.step();
//
// | idx | history    | t |
// |-----|------------|---|
// |  0  | {11000000} | 2 |
// |  1  | {00110000} | 0 | <-- curr_idx
// |  2  | {00001100} | 1 |
// =============================================================================
void BlockOutput::step() {

    curr_idx++;

    if (curr_idx > (uint32_t)history.size() - 1)
        curr_idx = 0;
}
// =============================================================================
// # Store
//
// Copy state BitArray into the current element of the history vector.
//
// ## Example
//
// | idx | history    | t |
// |-----|------------|---|
// |  0  | {11000000} | 2 |
// |  1  | {00110000} | 0 | <-- curr_idx
// |  2  | {00001100} | 1 |
//
// out.store();
//
//  state: {00000011} ----------------------
//                                         |
// | idx | history    | t |                |
// |-----|------------|---|                |
// |  0  | {11000000} | 2 |                |
// |  1  | {00000011} | 0 | <-- curr_idx ---
// |  2  | {00001100} | 1 |
// =============================================================================
void BlockOutput::store() {

    // Update changed_flag
    changed_flag = state != history[idx(PREV)];

    // Copy state into the current element of the history vector.
    history[curr_idx] = state;
    changes[curr_idx] = changed_flag;
}

// =============================================================================
// # Memory Usage
//
// Returns an estimate of the number of bytes used by the BlockOutput.
// =============================================================================
uint32_t BlockOutput::memory_usage() {

    uint32_t bytes = 0;
    uint32_t num_t = (uint32_t)history.size();

    bytes += state.memory_usage();
    bytes += sizeof(id);
    bytes += sizeof(curr_idx);
    bytes += sizeof(changed_flag);
    bytes += num_t * history[0].memory_usage();
    bytes += num_t * sizeof(changes[0]);

    return bytes;
}

// =============================================================================
// # Idx
//
// Get history index based on time step.
//
// TODO: figure out a way to return uint32_t instead of an int
//
// ## Example
//
// | idx | history    | t |
// |-----|------------|---|
// |  0  | {11000000} | 2 |
// |  1  | {00000011} | 0 | <-- curr_idx
// |  2  | {00001100} | 1 |
//
// - idx(0) returns 1 (the index of the current bitarray)
// - idx(1) returns 2 (the index of the previous bitarray)
// - idx(2) returns 0 (the index of the bitarray two time steps ago)
// =============================================================================
int BlockOutput::idx(const int ts) {

    assert(ts >= 0 && ts < (int)history.size());

    int i = curr_idx - ts;

    if (i < 0)
        i += (int)history.size();

    return i;
}
