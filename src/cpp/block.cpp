// =============================================================================
// block.cpp
// =============================================================================
#include "block.hpp"

using namespace BrainBlocks;

uint32_t Block::next_id = 0;

// =============================================================================
// # Constructor
//
// Constructs a Block.
// =============================================================================
Block::Block(uint32_t seed) {

    this->id = next_id++;
    rng = std::mt19937(seed);
}

// =============================================================================
// # Initialize
//
// Initializes BlockMemories based on BlockInput parameters.
// =============================================================================
void Block::init() {

    init_flag = true;
}

// =============================================================================
// # Save
//
// Saves block memories.
// =============================================================================
bool Block::save(const char* file) {
    
    return true;
}

// =============================================================================
// # Load
//
// Loads block memories.
// =============================================================================
bool Block::load(const char* file) {
    
    return true;
}

// =============================================================================
// # Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void Block::clear() {}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void Block::step() {}

// =============================================================================
// # Pull
//
// Updates BlockInput state(s) from child BlockOutput histories.
// =============================================================================
void Block::pull() {}

// =============================================================================
// # Push
//
// Updates child BlockOutput state(s) from BlockInput state(s).
// =============================================================================
void Block::push() {}

// =============================================================================
// # Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void Block::encode() {}

// =============================================================================
// # Decode
//
// Converts BlockOutput state(s) into BlockInput state(s).
// =============================================================================
void Block::decode() {}

// =============================================================================
// # Learn
//
// Updates BlockMemories.
// =============================================================================
void Block::learn() {}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void Block::store() {}

// =============================================================================
// # Memory Usage
//
// Returns an estimate of the number of bytes used by the block.
// =============================================================================
uint32_t Block::memory_usage() {

    uint32_t bytes = 0;

    bytes += sizeof(id);
    bytes += sizeof(init_flag);

    return bytes;
}

// =============================================================================
// # Feedforward
//
// Performs all functions required to produce output from input.
// =============================================================================
void Block::feedforward(bool learn_flag) {

    if (!init_flag)
        init();

    step();
    pull();
    encode();
    store();

    if (learn_flag)
        learn();
}

// =============================================================================
// # Feedback
//
// Performs all functions required to procuce input from output.
// =============================================================================
void Block::feedback() {

    if (!init_flag)
        init();

    decode();
    push();
}
