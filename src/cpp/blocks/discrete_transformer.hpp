// =============================================================================
// discrete_transformer.hpp
// =============================================================================
#ifndef DISCRETE_TRANSFORMER_HPP
#define DISCRETE_TRANSFORMER_HPP

#include "../block.hpp"
#include "../block_output.hpp"

namespace BrainBlocks {

class DiscreteTransformer final : public Block {

public:

    // Constructor
    DiscreteTransformer(
        const uint32_t num_v,
        const uint32_t num_s,
        const uint32_t num_t=2,
        const uint32_t seed=0);

    // Overrided virtual functions
    void clear() override;
    void step() override;
    void encode() override;
    void decode() override;
    void store() override;

    // Getters and setters
    void set_value(const uint32_t val) { value = val; };
    uint32_t get_value() { return value; };

    // Block IO variables
    BlockOutput output;

private:

    uint32_t value = 0;
    uint32_t value_prev = 0xFFFFFFFF;
    uint32_t num_v; // number of discrete values
    uint32_t num_s;  // number of statelets
    uint32_t num_as; // number of active statelets
    uint32_t dif_s;  // num_s - num_as
};

} // namespace BrainBlocks

#endif // DISCRETE_TRANSFORMER_HPP
