// =============================================================================
// block_input.cpp
// =============================================================================
#include "block_input.hpp"
#include <cassert>

using namespace BrainBlocks;

uint32_t BlockInput::next_id = 0;

// =============================================================================
// # Constructor
//
// Constructs a BlockInput.
// =============================================================================
BlockInput::BlockInput() {

    this->id = next_id++;
}

// =============================================================================
// # Add Child
//
// Connects a BlockOutput at a prior time step to the BlockInput.
//
// ## Example
//
// BlockOutput output0;
// BlockOutput output1;
// ...
// BlockInput input;
//
// output0.setup(4, 128); // Contains 4 time steps and 128 bits per BitArray
// output1.setup(2, 512); // Contains 2 time steps and 512 bits per BitArray
// ...
//
// input.add_child(&output0, 2); // Connect to output0 2 time steps ago
// input.add_child(&output1, 0); // Connect to output1 at the current time step
// ...
//
// Resulting BlockInput Children
// -----------------------------
//     children: {&output0, &output1, ...}
//        times: {       2,        0, ...}
// word_offsets: {       0,        4, ...}
//   word_sizes: {       4,       16, ...}
// =============================================================================
void BlockInput::add_child(BlockOutput* src, uint32_t src_t) {

    assert(src != nullptr);
    assert(src_t < src->num_t());

    uint32_t num_c = (uint32_t)children.size();
    uint32_t word_offset = 0;
    uint32_t word_size = src->state.num_words();
    uint32_t num_bits = state.num_bits() + word_size * 32;

    if (num_c > 0) {
        uint32_t j = num_c - 1;
        word_offset = word_offsets[j] + word_sizes[j];
    }

    children.push_back(src);
    times.push_back(src_t);
    word_offsets.push_back(word_offset);
    word_sizes.push_back(word_size);

    state.resize(num_bits);
}

// =============================================================================
// # Clear
//
// Assign all bits to 0 in the state BitArray.
// =============================================================================
void BlockInput::clear() {

    state.clear_all();
}

// =============================================================================
// # Pull
//
// Update state BitArray from connected child BlockOutput BitArray histories.
//
// ## Example
//
// TODO: Bit-indices are used for offset and size in this example when the
//       implementation actually uses word-indices.  The bitarray_copy()
//       function needs to be modified to work on bit-indices.
//
// input.pull();
//
// input
// ==========
//   size[1]: 12 --------------------------------+
// offset[1]:  8 ---------------------+          |
//   size[0]:  8 -------------------+ |          |
// offset[0]:  0 ------------+      | |          |
//                           |      | |          |
//                           v      v v          v
//             input.state: {11000000 000011110000 ...}
//                           ^        ^
// children[0]               |        |
// =============             |        |
//        state: {00000000}  |        |
// history[t=0]: {11000000}--+        |
// history[t=1]: {00110000}           |
// history[t=2]: {00001100}           |
// history[t=3]: {00000011}           |
//                                    |
// children[1]                        |
// =============                      |
//        state: {000000000000}       |
// history[t=2]: {000000001111}       |
// history[t=0]: {111100000000}       |
// history[t=1]: {000011110000}-------+
// =============================================================================
void BlockInput::pull() {

    for (uint32_t c = 0; c < children.size(); c++) {
    BitArray* child  = &children[c]->get_bitarray(times[c]);
        bitarray_copy(&state, child, word_offsets[c], 0, word_sizes[c]);
    }
}

// =============================================================================
// # Push
//
// Update connected child BlockOutput state BitArray from state BitArray.
//
// ## Example
//
// TODO: Bit-indices are used for offset and size in this example when the
//       implementation actually uses word-indices.  The bitarray_copy()
//       function needs to be modified to work on bit-indices.
//
// input.push();
//
// input
// ==========
//   size[1]: 12 --------------------------------+
// offset[1]:  8 ---------------------+          |
//   size[0]:  8 -------------------+ |          |
// offset[0]:  0 ------------+      | |          |
//                           |      | |          |
//                           v      v v          v
//             input.state: {10001110 110011001100 ...}
//                           |        |
// children[0]               |        |
// =============             |        |
//        state: {10001110}<-+        |
// history[t=0]: {11000000}           |
// history[t=1]: {00110000}           |
// history[t=2]: {00001100}           |
// history[t=3]: {00000011}           |
//                                    |
// children[1]                        |
// =============                      |
//        state: {110011001100}<------+
// history[t=0]: {000000001111}
// history[t=0]: {111100000000}
// history[t=0]: {000011110000}
// =============================================================================
void BlockInput::push() {

    for (uint32_t c = 0; c < children.size(); c++) {
        BitArray* child  = &children[c]->state;
        bitarray_copy(child, &state, 0, word_offsets[c], word_sizes[c]);
    }
}

// =============================================================================
// # Children Changed
//
// Returns true if children histories have changed.
// =============================================================================
bool BlockInput::children_changed() {

    bool changed = false;

    for (uint32_t c = 0; c < children.size(); c++) {
        if (children[c]->has_changed(times[c])) {
            changed = true;
            break;
        }
    }

    return changed;
}

// =============================================================================
// # Memory Usage
//
// Returns an estimate of the number of bytes used by the BlockInput.
// =============================================================================
uint32_t BlockInput::memory_usage() {

    uint32_t bytes = 0;
    uint32_t num_c = (uint32_t)children.size();

    bytes += state.memory_usage();
    bytes += sizeof(id);
    bytes += num_c * sizeof(children[0]);
    bytes += num_c * sizeof(times[0]);
    bytes += num_c * sizeof(word_offsets[0]);
    bytes += num_c * sizeof(word_sizes[0]);

    return bytes;
}
