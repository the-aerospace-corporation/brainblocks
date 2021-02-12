// =============================================================================
// _template.cpp
// =============================================================================
#include "_template.hpp"
//#include "../utils.hpp"
//#include <cassert>
//#include <iostream>

using namespace BrainBlocks;

// =============================================================================
// # Template
//
// Description of the Template class.
// =============================================================================

// =============================================================================
// # Constructor
//
// Constructs a Template.
// =============================================================================
Template::Template(
        //const uint32_t param0,
        //const uint32_t param1,
        //...
    ) : Block() {

    // Check parameters
    //assert(param0 > 0);
    //assert(param1 > 0);
    //...

    // Setup variables
    //this->param0 = param0;
    //this->param1 = param1;
    //this->param2 = param0 * param1;
    //...

    // Setup output
    //output.setup(num_t, num_s);
}

// =============================================================================
// # Initialize
//
// Initializes BlockMemories based on BlockInput parameters.
// =============================================================================
void Template::init() {

    //assert(whatever);

    //memory.init(params...);

    init_flag = true;
}

// =============================================================================
// Save
//
// Saves block memories.
// =============================================================================
void Template::save(const char* file) {

    //memory.save(file);
}

// =============================================================================
// Load
//
// Loads block memories.
// =============================================================================
void Template::load(const char* file) {

    //memory.load(file)
}

// =============================================================================
// Clear
//
// Clears BlockInput, BlockMemory, and BlockOutput states.
// =============================================================================
void Template::clear() {

    //input.clear();
    //memory.clear();
    //output.clear();
}

// =============================================================================
// Step
//
// Updates BlockOutput history current index.
// =============================================================================
void Template::step() {

    //output.step();
}

// =============================================================================
// Pull
//
// Updates BlockInput state(s) from child BlockOutput histories.
// =============================================================================
void Template::pull() {

    //input.pull();
}

// =============================================================================
// Push
//
// Updates child BlockOutput state(s) from BlockInput state(s).
// =============================================================================
void Template::push() {

    //input.push();
}

// =============================================================================
// Encode
//
// Converts BlockInput state(s) into BlockOutput state(s).
// =============================================================================
void Template::encode() {

    if (!init_flag)
        init();

    //output.state.clear_all();
    //memory.state.set_bit(0);
    //output.state.set_bit(0);
}

// =============================================================================
// Decode
//
// Converts BlockOutput state(s) into BlockInput state(s).
// =============================================================================
void Template::decode() {

    if (!init_flag)
        init();

    //input.state.clear_all();
    //input.state.set_bit(1);
}

// =============================================================================
// Learn
//
// Updates BlockMemories.
// =============================================================================
void Template::learn() {

    if (!init_flag)
        init();

    /*
    std::vector<uint32_t> output_acts = output.state.get_acts();

    for (uint32_t i = 0; i < output_acts.size(); i++) {
        uint32_t d = output_acts[i];

        memory.learn(d, input.state);
    }
    */
}

// =============================================================================
// Store
//
// Copy BlockOutput state into current index of BlockOutput history.
// =============================================================================
void Template::store() {

    //output.store();
}

// =============================================================================
// Memory Usage
//
// Returns an estimate of the number of bytes used by the block.
// =============================================================================
uint32_t Template::memory_usage() {

    uint32_t bytes = 0;

    //bytes += input.memory_usage();
    //bytes += output.memory_usage();
    //bytes += memory.memory_usage();
    //bytes += sizeof(param0);
    //bytes += sizeof(param1);
    //bytes += sizeof(param2);
    //...

    return bytes;
}


// =============================================================================
// Public Function
// =============================================================================
//void Template::public_function() {}

// =============================================================================
// Private Function
// =============================================================================
//void Template::private_function() {}
