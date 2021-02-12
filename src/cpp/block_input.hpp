// =============================================================================
// block_input.hpp
// =============================================================================
#ifndef BLOCK_INPUT_HPP
#define BLOCK_INPUT_HPP

#include "bitarray.hpp"
#include "block_output.hpp"
#include <cstdint>
#include <vector>

namespace BrainBlocks {

class BlockInput {

public:

    BlockInput();

    void add_child(BlockOutput* src, uint32_t src_t);
    void clear();
    void pull();
    void push();
    bool children_changed();
    uint32_t memory_usage();

    uint32_t num_children() { return (uint32_t)children.size(); };

    BitArray state;

private:

    static uint32_t next_id;
    uint32_t id = 0xffffffff;

    // Child connection vectors
    std::vector<BlockOutput*> children;
    std::vector<uint32_t> times;
    std::vector<uint32_t> word_offsets;
    std::vector<uint32_t> word_sizes;
};

} // namespace BrainBlocks

#endif // BLOCK_INPUT_HPP
