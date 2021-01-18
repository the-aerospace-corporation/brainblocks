#include "blocks/context_learner.hpp"
#include "blocks/scalar_transformer.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    ScalarTransformer st0(0.0, 1.0, 64, 8, 2);
    ScalarTransformer st1(0.0, 1.0, 64, 8, 2);
    ContextLearner cl(64, 10, 10, 12, 6, 20, 2, 1, 2);

    cl.input.add_child(&st0.output, CURR);
    cl.context.add_child(&st1.output, CURR);

    t0 = std::chrono::high_resolution_clock::now();
    cl.init();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;

    st0.set_value(0.0);
    st1.set_value(0.0);
    st0.feedforward();
    st1.feedforward();
    cl.feedforward(true);
    cl.output[CURR].print_acts();

    st0.set_value(1.0);
    st1.set_value(0.0);
    st0.feedforward();
    st1.feedforward();
    cl.feedforward(true);
    cl.output[CURR].print_acts();

    st0.set_value(0.0);
    st1.set_value(1.0);
    st0.feedforward();
    st1.feedforward();
    cl.feedforward(true);
    cl.output[CURR].print_acts();

    st0.set_value(1.0);
    st1.set_value(1.0);
    st0.feedforward();
    st1.feedforward();
    cl.feedforward(true);
    cl.output[CURR].print_acts();

    return 0;
}
