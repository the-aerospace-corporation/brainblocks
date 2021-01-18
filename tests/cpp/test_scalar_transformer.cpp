// =============================================================================
// test_scalar_transformer.cpp
// =============================================================================
#include "blocks/scalar_transformer.hpp"
#include <iostream>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    const uint32_t NUM_BITS = 1024;
    const uint32_t NUM_ACTS =  128;

    std::cout << "ScalarTransformer st(...);" << std::endl;
    std::cout << "--------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ScalarTransformer st(0.0, 1.0, NUM_BITS, NUM_ACTS, 3);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; st.output.state.print_acts();
    std::cout << "size=" << st.output.state.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "st.set_value(0.0);" << std::endl;
    std::cout << "st.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    st.set_value(0.0);
    st.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; st.output[0].print_acts();
    std::cout << "acts[1]="; st.output[1].print_acts();
    std::cout << "acts[2]="; st.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "st.set_value(0.5);" << std::endl;
    std::cout << "st.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    st.set_value(0.5);
    st.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; st.output[0].print_acts();
    std::cout << "acts[1]="; st.output[1].print_acts();
    std::cout << "acts[2]="; st.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "st.set_value(1.0);" << std::endl;
    std::cout << "st.feedforward(); " << std::endl;
    std::cout << "-----------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    st.set_value(1.0);
    st.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; st.output[0].print_acts();
    std::cout << "acts[1]="; st.output[1].print_acts();
    std::cout << "acts[2]="; st.output[2].print_acts();
    std::cout << std::endl;

    return 0;
}
