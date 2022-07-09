// =============================================================================
// blank_block.hpp
// =============================================================================
#ifndef BLANKBLOCK_HPP
#define BLANKBLOCK_HPP

#include "../block.hpp"
#include "../block_output.hpp"

namespace BrainBlocks {

class BlankBlock final : public Block {

public:

    // Constructor
    BlankBlock(
        const uint32_t num_s,
        const uint32_t num_t=2,
        const uint32_t seed=0);

    // Overrided virtual functions
    void clear() override;
    void step() override;
    void store() override;
    uint32_t memory_usage() override;

    // Block IO and Memory
    BlockOutput output;
};

} // namespace BrainBlocks

#endif // BLANKBLOCK_HPP
