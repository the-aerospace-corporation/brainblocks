#ifndef PATTERN_POOLER_HPP
#define PATTERN_POOLER_HPP

#define _CRT_SECURE_NO_WARNINGS // remove Windows fopen warnings

#include "bitarray.hpp"
#include "page.hpp"
#include "coincidence_set.hpp"
#include <cstdint>
#include <vector>

class PatternPooler {
    public:
        PatternPooler(
            const uint32_t num_s,
            const uint32_t num_as,
            const uint8_t perm_thr,
            const uint8_t perm_inc,
            const uint8_t perm_dec,
            const double pct_pool,
            const double pct_conn,
            const double pct_learn);

        void initialize();
        void save(const char* file);
        void load(const char* file);
        void clear_states();
        void compute(const uint32_t learn_flag = false);
        Page& get_input() { return input; };
        Page& get_output() { return output; };
        CoincidenceSet& get_output_coincidence_set(const uint32_t d) { return d_output[d]; };

    private:
        void overlap();
        void activate();
        void learn();

    private:
        uint32_t num_s;       // number of statelets
        uint32_t num_as;      // number of active statelets
        uint8_t perm_thr;    // permanence threshold
        uint8_t perm_inc;    // permanence increment
        uint8_t perm_dec;    // permanence decrement
        double pct_pool;      // percent pool
        double pct_conn;      // percent initially connected
        double pct_learn;     // percent learn
        bool init_flag;    // initialized flag
        BitArray lmask_ba; // learning mask bitarray
        std::vector<uint32_t> d_output_overlaps; // output coincidence detector overlap scores
        std::vector<uint32_t> d_output_templaps; // output coincidence detector temporary overlap scores
        std::vector<CoincidenceSet> d_output; // output coincidence detectors
        Page input;  // input page object
        Page output; // output page object
};

#endif