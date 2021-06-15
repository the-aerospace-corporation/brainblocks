#include "blocks/pattern_pooler.hpp"
#include "blocks/scalar_transformer.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    std::vector<double> values = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    ScalarTransformer st(0.0, 1.0, 1024, 8);
    PatternPooler pp(1024, 8, 20, 2, 1, 0.8, 0.5, 0.3, 2);

    pp.input.add_child(&st.output, 0);

    pp.init();

    for (uint32_t i = 0; i < values.size(); i++) {
        st.set_value(values[i]);
        st.feedforward();

        t0 = std::chrono::high_resolution_clock::now();
        pp.feedforward(true);
        t1 = std::chrono::high_resolution_clock::now();
        duration = t1 - t0;
        std::cout << "t=" << duration.count() << "s" << std::endl;

        //e.output[CURR].print_bits();
        //pp.output[CURR].print_bits();
        //std::cout << std::endl;
    }

    return 0;
}
