#ifndef COINCIDENCE_SET_HPP
#define COINCIDENCE_SET_HPP

#include "bitarray.hpp"
#include <cstdint>
#include <vector>

#define PERM_MIN 0
#define PERM_MAX 99

class CoincidenceSet {
    public:
        CoincidenceSet() {};
        void resize(const uint32_t num_r);

        void initialize_pool(
            const uint32_t num_r,    // number of receptors
            const uint32_t num_i,    // number of input bits
            const uint32_t num_conn, // number of initially connected receptors
            const uint8_t perm_thr); // permanence threshold

        uint32_t overlap(BitArray& input_ba, const uint8_t perm_thr);

        void learn(
            BitArray& input_ba,
            BitArray& lmask_ba,
            const uint8_t perm_inc,
            const uint8_t perm_dec);

        void learn_move(
            BitArray& input_ba,
            BitArray& lmask_ba,
            const uint8_t perm_inc,
            const uint8_t perm_dec);

        void punish(
            BitArray& input_ba,
            BitArray& lmask_ba,
            const uint8_t perm_inc);

        void print_addrs();
        void print_perms();
        
        uint32_t get_num_r() { return (uint32_t)addrs.size(); };

    public: //TODO: make private?
        std::vector<uint32_t> addrs; // receptor addresses
        std::vector<uint8_t> perms;  // receptor permanences
};

#endif