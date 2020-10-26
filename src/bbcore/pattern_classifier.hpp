#ifndef PATTERN_CLASSIFIER_HPP
#define PATTERN_CLASSIFIER_HPP

#define _CRT_SECURE_NO_WARNINGS // remove Windows fopen warnings

#include "bitarray.hpp"
#include "page.hpp"
#include "coincidence_set.hpp"
#include <stdint.h>

class PatternClassifier {
    public:
        PatternClassifier(
            const uint32_t* labels,
            const uint32_t num_l,
            const uint32_t num_s,
            const uint32_t num_as,
            const uint32_t perm_thr,
            const uint32_t perm_inc,
            const uint32_t perm_dec,
            const double pct_pool,
            const double pct_conn,
            const double pct_learn);

        ~PatternClassifier();

        void initialize();
        void save(const char* file);
        void load(const char* file);
        void clear();
        void compute(const uint32_t in_label, const uint32_t learn_flag);
        void update_probabilities();
        void overlap();
        void activate();
        void learn(const uint32_t in_label);
        struct BitArray* decode();

    private:
        uint32_t num_l;       // number of labels
        uint32_t num_s;       // number of statelets
        uint32_t num_as;      // number of active statelets
        uint32_t num_spl;     // number of states per label
        uint32_t perm_thr;    // permanence threshold
        uint32_t perm_inc;    // permanence increment
        uint32_t perm_dec;    // permanence decrement
        double pct_pool;      // percent pool
        double pct_conn;      // percent initially connected
        double pct_learn;     // percent learn
        uint8_t init_flag;    // initialized flag
        uint32_t* s_labels;   // statelet labels
        uint32_t* learn_mask; // receptor random learning mask array
        uint32_t* labels;     // labels
        double* probs;        // probabilities array
        struct Page* input;   // input page object
        struct Page* output;  // output page object
        struct CoincidenceSet* coincidence_sets;
};

#endif