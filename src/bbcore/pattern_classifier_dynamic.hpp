#ifndef PATTERN_CLASSIFIER_DYNAMIC_HPP
#define PATTERN_CLASSIFIER_DYNAMIC_HPP

#define _CRT_SECURE_NO_WARNINGS // remove Windows fopen warnings

#include "bitarray.hpp"
#include "page.hpp"
#include "coincidence_set.hpp"
#include <cstdint>
#include <vector>

class PatternClassifierDynamic {
    public:
        PatternClassifierDynamic(
            const uint32_t num_s,
            const uint32_t num_as,
            const uint32_t num_spl,
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
        void compute(const uint32_t label = 0xFFFFFFFF, const uint32_t learn_flag = false);
        //struct BitArray* decode(); // TODO: add decoding to each block
        std::vector<uint32_t> get_labels() { return labels; };
        std::vector<double> get_probabilities() { return probs; };
        double get_score() { return pct_score; };
        Page& get_input() { return input; };
        Page& get_output() { return output; };
        CoincidenceSet& get_output_coincidence_set(const uint32_t d) { return d_output[d]; };

    private:
        void overlap();
        void activate();
        void learn(const uint32_t label);

    private:
        uint32_t num_s;    // number of statelets
        uint32_t num_as;   // number of active statelets
        uint32_t num_spl;  // number of states per label
        uint8_t perm_thr;  // permanence threshold
        uint8_t perm_inc;  // permanence increment
        uint8_t perm_dec;  // permanence decrement
        double pct_pool;   // percent pool
        double pct_conn;   // percent initially connected
        double pct_learn;  // percent learn
        double pct_score;  // abnormality score
        bool init_flag;    // initialized flag
        BitArray lmask_ba; // learning mask bitarray
        std::vector<BitArray> l_states; // array of bitarrays assigned by label that hold which statelets are assigned to that label
        std::vector<uint32_t> labels;   // labels array
        std::vector<uint32_t> counts;   // counts array
        std::vector<double> probs;      // probabilities array
        std::vector<uint32_t> d_output_overlaps; // output coincidence detector overlap scores
        std::vector<uint32_t> d_output_templaps; // output coincidence detector temporary overlap scores
        std::vector<CoincidenceSet> d_output; // output coincidence detectors
        Page input;  // input page object
        Page output; // output page object
};

#endif