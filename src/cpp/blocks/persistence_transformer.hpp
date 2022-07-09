// =============================================================================
// persistence_transformer.hpp
// =============================================================================
#ifndef PERSISTENCE_TRANSFORMER_HPP
#define PERSISTENCE_TRANSFORMER_HPP

#include "../block.hpp"
#include "../block_output.hpp"

namespace BrainBlocks {

class PersistenceTransformer final : public Block {

public:

    // Constructor
    PersistenceTransformer(
        const double min_val,
        const double max_val,
        const uint32_t num_s,
        const uint32_t num_as,
        const uint32_t max_step,
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
    double min_val;  // maximum input value
    double max_val;  // minimum input value
    double dif_val;  // max_val - min_val
    uint32_t num_s;  // number of statelets
    uint32_t num_as; // number of active statelets
    uint32_t dif_s;  // num_s - num_as
    uint32_t counter;    // step counter
    uint32_t max_step;  // maximum steps
    double pct_val_prev; // previous percentage
};

} // namespace BrainBlocks

#endif // PERSISTENCE_TRANSFORMER_HPP
