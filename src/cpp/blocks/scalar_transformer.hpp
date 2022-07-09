// =============================================================================
// scalar_transformer.hpp
// =============================================================================
#ifndef SCALAR_TRANSFORMER_HPP
#define SCALAR_TRANSFORMER_HPP

#include "../block.hpp"
#include "../block_output.hpp"

namespace BrainBlocks {

class ScalarTransformer final : public Block {

public:

    // Constructor
    ScalarTransformer(
        const double min_val,
        const double max_val,
        const uint32_t num_s,
        const uint32_t num_as,
        const uint32_t num_t=2,
        const uint32_t seed=0);

    // Overrided virtual functions
    void clear() override;
    void step() override;
    void encode() override;
    void decode() override;
    void store() override;

    // Getters and setters
    void set_value(const double val) { value = val; };
    double get_value() { return value; };

    // Block IO variables
    BlockOutput output;

private:

    double value = 0.0;
    double value_prev = 0.123456789;
    double min_val;  // maximum input value
    double max_val;  // minimum input value
    double dif_val;  // max_val - min_val
    uint32_t num_s;  // number of statelets
    uint32_t num_as; // number of active statelets
    uint32_t dif_s;  // num_s - num_as
};

} // namespace BrainBlocks

#endif // SCALAR_TRANSFORMER_HPP
