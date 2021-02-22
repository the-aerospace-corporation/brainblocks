// =============================================================================
// test_discrete_transformer.cpp
// =============================================================================
#include "blocks/discrete_transformer.hpp"
#include <iostream>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    const uint32_t NUM_BITS = 1024;

    std::cout << "DiscreteTransformer dt(...);" << std::endl;
    std::cout << "-------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    DiscreteTransformer dt(16, NUM_BITS, 3);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; dt.output.state.print_acts();
    std::cout << "size=" << dt.output.state.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "dt.set_value(0);" << std::endl;
    std::cout << "dt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    dt.set_value(0);
    dt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; dt.output[0].print_acts();
    std::cout << "acts[1]="; dt.output[1].print_acts();
    std::cout << "acts[2]="; dt.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "dt.set_value(1);" << std::endl;
    std::cout << "dt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    dt.set_value(1);
    dt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; dt.output[0].print_acts();
    std::cout << "acts[1]="; dt.output[1].print_acts();
    std::cout << "acts[2]="; dt.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "dt.set_value(2);" << std::endl;
    std::cout << "dt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    dt.set_value(2);
    dt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; dt.output[0].print_acts();
    std::cout << "acts[1]="; dt.output[1].print_acts();
    std::cout << "acts[2]="; dt.output[2].print_acts();
    std::cout << std::endl;

    return 0;
}
