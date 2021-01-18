// =============================================================================
// block_memory.hpp
// =============================================================================
#ifndef BLOCK_MEMORY_HPP
#define BLOCK_MEMORY_HPP

#include "bitarray.hpp"
#include <cstdint>
#include <vector>
#include <random>

#define PERM_MIN 0
#define PERM_MAX 99

namespace BrainBlocks {

class BlockMemory {

public:

    // Initializers
    void init(
        const uint32_t num_i,
        const uint32_t num_d,
        const uint32_t num_rpd,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec,
        const double pct_learn);

    void init_conn(
        const uint32_t num_i,
        const uint32_t num_d,
        const uint32_t num_rpd,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec,
        const double pct_learn);

    void init_pooled(
        const uint32_t num_i,
        const uint32_t num_d,
        const double pct_pool,
        const double pct_conn,
        const double pct_learn,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec,
        std::mt19937& rng);

    void init_pooled_conn(
        const uint32_t num_i,
        const uint32_t num_d,
        const double pct_pool,
        const double pct_conn,
        const double pct_learn,
        const uint8_t perm_thr,
        const uint8_t perm_inc,
        const uint8_t perm_dec,
        std::mt19937& rng);

    // Misc. functions
    void save(FILE* fptr);
    void load(FILE* fptr);
    void clear();
    uint32_t memory_usage();

    // Core functions
    uint32_t overlap(
        const uint32_t d,
        BitArray& input);

    uint32_t overlap_conn(
        const uint32_t d,
        BitArray& input);

    void learn(
        const uint32_t d,
        BitArray& input,
        std::mt19937& rng);

    void learn_conn(
        const uint32_t d,
        BitArray& input,
        std::mt19937& rng);

    void learn_move(
        const uint32_t d,
        BitArray& input,
        std::mt19937& rng);

    void learn_move_conn(
        const uint32_t d,
        BitArray& input,
        std::mt19937& rng);

    void punish(
        const uint32_t d,
        BitArray& input,
        std::mt19937& rng);

    void punish_conn(
        const uint32_t d,
        BitArray& input,
        std::mt19937& rng);

    // Printers
    void print_addrs(const uint32_t d);
    void print_perms(const uint32_t d);
    void print_conns(const uint32_t d);

    // Getters
    std::vector<uint32_t> addrs(const uint32_t d);
    std::vector<uint8_t> perms(const uint32_t d);
    std::vector<uint8_t> conns(const uint32_t d);
    uint32_t num_dendrites() { return num_d; };

    // Dendrite activations (0=inactive, 1=active)
    BitArray state;

private:

    void update_conns(const uint32_t d);

    // Flags
    bool init_flag = false;
    bool conns_flag = false;

    // Parameters
    uint32_t num_i;   // number of inputs
    uint32_t num_d;   // number of dendrites
    uint32_t num_rpd; // number of receptors per dendrite
    uint32_t num_r;   // number of receptors
    uint8_t perm_thr; // receptor permanence threshold
    uint8_t perm_inc; // receptor permanence increment
    uint8_t perm_dec; // receptor permanence decrement
    double pct_learn; // learning percentage

    // Arrays
    std::vector<uint32_t> r_addrs; // receptor addresses
    std::vector<uint8_t>  r_perms; // receptor permancences
    std::vector<BitArray> d_conns; // dendrite connections (optional)
    BitArray lmask;                // learning mask
};

} // namespace BrainBlocks

#endif // BLOCK_MEMORY_HPP
