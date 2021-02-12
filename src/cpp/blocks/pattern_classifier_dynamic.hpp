// =============================================================================
// pattern_classifier_dynamic.hpp
// =============================================================================
#ifndef PATTERN_CLASSIFIER_DYNAMIC_HPP
#define PATTERN_CLASSIFIER_DYNAMIC_HPP

#include "../block.hpp"
#include "../block_input.hpp"
#include "../block_memory.hpp"
#include "../block_output.hpp"

#include <vector>

namespace BrainBlocks {

class PatternClassifierDynamic final : public Block {

public:

    // Constructor
    PatternClassifierDynamic(
        const uint32_t num_s,
        const uint32_t num_as,
        const uint32_t num_spl,
        const uint8_t perm_thr=20,
        const uint8_t perm_inc=2,
        const uint8_t perm_dec=1,
        const double pct_pool=0.8,
        const double pct_conn=0.5,
        const double pct_learn=0.3,
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

    // Setters
    void set_label(const uint32_t label) { this->label = label; };

    // Getters
    double get_anomaly_score() { return pct_anom; };
    std::vector<uint32_t> get_labels() { return labels; };
    std::vector<double> get_probabilities();

    // Block IO and memory variables
    BlockInput input;
    BlockOutput output;
    BlockMemory memory;

private:

    uint32_t label;   // input label
    uint32_t num_s;   // number of statelets
    uint32_t num_as;  // number of active statelets
    uint32_t num_spl; // number of statelets per label
    uint8_t perm_thr; // permanence threshold
    uint8_t perm_inc; // permanence increment
    uint8_t perm_dec; // permanence decrement
    double pct_pool;  // percent pooled
    double pct_conn;  // percent initially connected
    double pct_learn; // percent learn
    double pct_anom;

    std::vector<uint32_t> overlaps; // overlaps
    std::vector<uint32_t> templaps; // temporary overlaps
    std::vector<uint32_t> labels;
    std::vector<uint32_t> counts;
    std::vector<BitArray> l_states;
};

} // namespace BrainBlocks

#endif // PATTERN_CLASSIFIER_DYNAMIC_HPP
