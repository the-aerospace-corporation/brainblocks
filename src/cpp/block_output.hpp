// =============================================================================
// block_output.hpp
// =============================================================================
#ifndef BLOCK_OUTPUT_HPP
#define BLOCK_OUTPUT_HPP

#include "bitarray.hpp"
#include <vector>
#include <cstdint>

#define CURR 0
#define PREV 1

namespace BrainBlocks {

class BlockOutput {

public:

    BlockOutput();

    void setup(const uint32_t num_t, const uint32_t num_b);
    void clear();
    void step();
    void store();
    uint32_t memory_usage();

    // Getters
    bool has_changed() { return changed_flag; };
    bool has_changed(const int t) { return changes[idx(t)]; };
    BitArray& get_bitarray(const int t) { return history[idx(t)]; };
    BitArray& operator[](const int t) { return history[idx(t)]; };
    uint32_t num_t() { return (uint32_t)history.size(); };

    // BlockOutput working BitArray
    BitArray state;

private:

    // Get history index based on time step
    int idx(const int ts);

    static uint32_t next_id;
    uint32_t id = 0xffffffff;
    uint32_t curr_idx = 0xffffffff;
    bool changed_flag = false;

    // History vectors
    std::vector<BitArray> history;
    std::vector<bool> changes;
};

} // namespace BrainBlocks

#endif // BLOCK_OUTPUT_HPP
