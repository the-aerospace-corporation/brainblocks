#ifndef PERSISTENCE_ENCODER_HPP
#define PERSISTENCE_ENCODER_HPP

#include "page.hpp"
#include <cstdint>

class PersistenceEncoder {
    public:
        PersistenceEncoder(
            const double min_val,
            const double max_val,
            const uint32_t num_s,
            const uint32_t num_as,
            const uint32_t max_steps);

        void initialize();
        void clear();
        void reset();
        void compute(double value);

    public:
        Page output; // output page object

    private:
        double min_val;      // minimum input value
        double max_val;      // maximum input value
        double range_val;    // value range
        uint32_t num_s;      // number of statelets
        uint32_t num_as;     // number of active statelets
        uint32_t range_bits; // bit range
        uint32_t max_steps;  // maximum steps
        uint32_t step;       // step counter
        double pct_val_prev; // value previous percentage
        bool init_flag;      // initialized flag
};

#endif