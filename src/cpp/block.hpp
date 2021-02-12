// =============================================================================
// block.hpp
// =============================================================================
#ifndef BLOCK_HPP
#define BLOCK_HPP

#include <cstdint>
#include <random>
#include <cstdio>
#include <cassert>

namespace BrainBlocks {

class Block {

public:

    // Constructor and destructor
    Block(uint32_t seed=0);
    ~Block() {}; // TODO: make virtual destructor

    // Virtual functions
    virtual void init();
    virtual bool save(const char* file);
    virtual bool load(const char* file);
    virtual void clear();
    virtual void step();
    virtual void pull();
    virtual void push();
    virtual void encode();
    virtual void decode();
    virtual void learn();
    virtual void store();
    virtual uint32_t memory_usage();

    // Public functions
    void feedforward(bool learn_flag=false);
    void feedback();

protected:

    static uint32_t next_id;
    uint32_t id = 0xffffffff;
    bool init_flag = false;
    std::mt19937 rng;
};

} // namespace BrainBlocks

#endif // BLOCK_HPP
