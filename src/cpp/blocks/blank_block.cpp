// =============================================================================
// blank_block.cpp
// =============================================================================
#include "blank_block.hpp"
#include <cassert>

using namespace BrainBlocks;

// =============================================================================
// # BlankBlock
//
// An empty block that can connect to other blocks.  Useful for custom ad-hoc
// block implementations created by end-users where they can transfer bit
// information to the BlankBlock output BitArray and have it represented within
// and user-defined block hierarchy.
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a BlankBlock.
// =============================================================================
BlankBlock::BlankBlock(
        const uint32_t num_s,
        const uint32_t num_t,
        const uint32_t seed     // seed for random number generator
    ) : Block(seed) {

    // Check parameters
    assert(num_s > 0);

    // Setup output
    output.setup(num_t, num_s);
}

// =============================================================================
// # Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void BlankBlock::clear() {

    output.clear();
}

// =============================================================================
// # Step
//
// Updates BlockOutput history current index.
// =============================================================================
void BlankBlock::step() {

    output.step();
}

// =============================================================================
// # Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void BlankBlock::store() {

    output.store();
}

// =============================================================================
// # Memory Usage
//
// Returns an estimate of the number of bytes used by the block.
// =============================================================================
uint32_t BlankBlock::memory_usage() {

    uint32_t bytes = 0;

    bytes += output.memory_usage();

    return bytes;
}
