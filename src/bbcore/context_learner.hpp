#ifndef CONTEXT_LEARNER_HPP
#define CONTEXT_LEARNER_HPP

#define _CRT_SECURE_NO_WARNINGS // remove Windows fopen warnings

#include "bitarray.hpp"
#include "page.hpp"
#include "coincidence_set.hpp"
#include <cstdint>
#include <vector>

class ContextLearner {
    public:
        ContextLearner(
            const uint32_t num_spc,
            const uint32_t num_dps,
            const uint32_t num_rpd,
            const uint32_t d_thresh,
            const uint8_t perm_thr,
            const uint8_t perm_inc,
            const uint8_t perm_dec);

        void initialize();
        void save(const char* file);
        void load(const char* file);
        void clear_states();
        void compute(const bool learn_flag = true);
        double get_score() { return pct_score; };
        Page& get_input() { return input; };
        Page& get_context() { return context; };
        Page& get_output() { return output; };
        CoincidenceSet& get_output_coincidence_set(const uint32_t d) { return d_output[d]; };

    private:
        void overlap();
        void activate(const uint32_t learn_flag);
        void learn();

    private:
        uint32_t num_c;    // number of columns
        uint32_t num_s;    // number of statelets
        uint32_t num_d;    // number of coincidence detectors
        uint32_t num_spc;  // number of statelets per column
        uint32_t num_dps;  // number of coincidence detectors per statelet
        uint32_t num_dpc;  // number of coincidence detectors per column
        uint32_t num_rpd;  // number of receptors per coincidence detector
        uint32_t d_thresh; // coincidence detector threshold
        uint8_t perm_thr;  // permanence increment
        uint8_t perm_inc;  // permanence increment
        uint8_t perm_dec;  // permanence decrement
        double pct_score;  // abnormality score percentage (0.0 to 1.0)
        bool init_flag;    // initialized flag
        std::vector<uint32_t> s_next_d; // array of next available coincidence detector index for each statelet
        std::vector<uint32_t> d_output_overlaps; // output coincidence detector overlap scores
        std::vector<CoincidenceSet> d_output; // array of output coincidence detectors
        BitArray d_output_states; // hidden coincidence detector states bitarray
        BitArray lmask_ba; // learning mask bitarray
        Page input;   // input page object
        Page context; // context page object
        Page output;  // output page object
};

#endif