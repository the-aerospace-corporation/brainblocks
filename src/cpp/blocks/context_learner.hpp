// =============================================================================
// context_learner.hpp
// =============================================================================
#ifndef CONTEXT_LEARNER_HPP
#define CONTEXT_LEARNER_HPP

#include "../block.hpp"
#include "../block_input.hpp"
#include "../block_memory.hpp"
#include "../block_output.hpp"

#include <vector>

namespace BrainBlocks {

class ContextLearner final : public Block {

public:

    // Constructor
    ContextLearner(
        const uint32_t num_c,
        const uint32_t num_spc,
        const uint32_t num_dps,
        const uint32_t num_rpd,
        const uint32_t d_thresh,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec,
        const uint32_t num_t=2,
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

    // Getters
    double get_anomaly_score() { return pct_anom; };

    // Block IO and memory variables
    BlockInput input;
    BlockInput context;
    BlockOutput output;
    BlockMemory memory;

private:

    void recognition(const uint32_t c);
    void surprise(const uint32_t c);
    void set_next_available_dendrite(const uint32_t s);

    uint32_t num_c;    // number of columns
    uint32_t num_spc;  // number of statelets per column
    uint32_t num_dps;  // number of dendrites per statelet
    uint32_t num_dpc;  // number of dendrites per column
    uint32_t num_rpd;  // number of receptors per dendrite
    uint32_t num_s;    // number of statelets
    uint32_t num_d;    // number of dendrites
    uint32_t d_thresh; // dendrite threshold
    uint8_t perm_thr;  // permanence threshold
    uint8_t perm_inc;  // permanence increment
    uint8_t perm_dec;  // permanence decrement
    double pct_anom; // anomaly score percentage (0.0 to 1.0)

    bool surprise_flag = false;
    std::vector<uint32_t> input_acts;
    std::vector<uint32_t> next_sd; // next available dendrite on statelets
    BitArray d_used; // (0 = dendrite available, 1 = dendrite in use)
};

} // namespace BrainBlocks

#endif // CONTEXT_LEARNER_HPP
