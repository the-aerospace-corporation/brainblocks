// =============================================================================
// test_label_transformer.cpp
// =============================================================================
#include "blocks/label_transformer.hpp"
#include <iostream>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    const uint32_t NUM_BITS = 1024;

    std::cout << "LabelTransformer lt(...);" << std::endl;
    std::cout << "-------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    LabelTransformer lt(16, NUM_BITS, 3);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; lt.output.state.print_acts();
    std::cout << "size=" << lt.output.state.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "lt.set_value(0);" << std::endl;
    std::cout << "lt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    lt.set_value(0);
    lt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; lt.output[0].print_acts();
    std::cout << "acts[1]="; lt.output[1].print_acts();
    std::cout << "acts[2]="; lt.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "lt.set_value(1);" << std::endl;
    std::cout << "lt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    lt.set_value(1);
    lt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; lt.output[0].print_acts();
    std::cout << "acts[1]="; lt.output[1].print_acts();
    std::cout << "acts[2]="; lt.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "lt.set_value(2);" << std::endl;
    std::cout << "lt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    lt.set_value(2);
    lt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; lt.output[0].print_acts();
    std::cout << "acts[1]="; lt.output[1].print_acts();
    std::cout << "acts[2]="; lt.output[2].print_acts();
    std::cout << std::endl;

    return 0;
}
