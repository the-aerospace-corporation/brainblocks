#ifndef SCALAR_ENCODER_HPP
#define SCALAR_ENCODER_HPP

#include "page.hpp"
#include <cstdint>

class ScalarEncoder {
    public:
        ScalarEncoder(
            const double min_val,
            const double max_val,
            const uint32_t num_s,
            const uint32_t num_as);

        void initialize();
        void clear_states();
        void compute(double value);
        Page& get_output() { return output; };

    private:
        double min_val;      // maximum input value
        double max_val;      // minimum input value
        double range_val;    // value range
        uint32_t num_s;      // number of statelets
        uint32_t num_as;     // number of active statelets
        uint32_t range_bits; // bit range
        bool init_flag;      // initialized flag
        Page output; // output page object
};

#endif