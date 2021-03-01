// =============================================================================
// pattern_pooler.hpp
// =============================================================================
#ifndef PATTERN_POOLER_HPP
#define PATTERN_POOLER_HPP

#include "../block.hpp"
#include "../block_input.hpp"
#include "../block_memory.hpp"
#include "../block_output.hpp"

#include <vector>

namespace BrainBlocks {

class PatternPooler final : public Block {

public:

    // Constructor
    PatternPooler(
        const uint32_t num_s,
        const uint32_t num_as,
        const uint8_t perm_thr=20,
        const uint8_t perm_inc=2,
        const uint8_t perm_dec=1,
        const double pct_pool=0.8,
        const double pct_conn=0.5,
        const double pct_learn=0.3,
        const uint32_t num_t=2,
        const bool always_update=false,
        const uint32_t seed=0);

    // Overrided functions
    void init() override;
    bool save(const char* file) override;
    bool load(const char* file) override;
    void clear() override;
    void step() override;
    void pull() override;
    // TODO: void push() override;
    void encode() override;
    // TODO: void decode() override;
    void learn() override;
    void store() override;
    // TODO: void bytes_used() override;

    // Block IO and memory variables
    BlockInput input;
    BlockOutput output;
    BlockMemory memory;

private:

    uint32_t num_s;   // number of statelets
    uint32_t num_as;  // number of active statelets
    uint8_t perm_thr; // permanence threshold
    uint8_t perm_inc; // permanence increment
    uint8_t perm_dec; // permanence decrement
    double pct_pool;  // percent pooled
    double pct_conn;  // percent initially connected
    double pct_learn; // percent learn
    bool always_update; // whether to only update on input changes

    std::vector<uint32_t> overlaps; // overlaps
    std::vector<uint32_t> templaps; // temporary overlaps
};

} // namespace BrainBlocks

#endif // PATTERN_POOLER_HPP
