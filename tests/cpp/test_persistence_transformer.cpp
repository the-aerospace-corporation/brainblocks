// =============================================================================
// test_persistence_transformer.cpp
// =============================================================================
#include "blocks/persistence_transformer.hpp"
#include <iostream>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    const uint32_t NUM_BITS = 1024;
    const uint32_t NUM_ACTS =  128;

    std::cout << "PersistenceTransformer pt(...);" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    PersistenceTransformer pt(0.0, 1.0, NUM_BITS, NUM_ACTS, 10, 3);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; pt.output.state.print_acts();
    std::cout << "size=" << pt.output.state.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "pt.set_value(0.0);" << std::endl;
    std::cout << "pt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    pt.set_value(0.0);
    pt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; pt.output[0].print_acts();
    std::cout << "acts[1]="; pt.output[1].print_acts();
    std::cout << "acts[2]="; pt.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "pt.set_value(0.0);" << std::endl;
    std::cout << "pt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    pt.set_value(0.0);
    pt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; pt.output[0].print_acts();
    std::cout << "acts[1]="; pt.output[1].print_acts();
    std::cout << "acts[2]="; pt.output[2].print_acts();
    std::cout << std::endl;

    std::cout << "pt.set_value(0.0);" << std::endl;
    std::cout << "pt.feedforward(); " << std::endl;
    std::cout << "------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    pt.set_value(0.0);
    pt.feedforward();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts[0]="; pt.output[0].print_acts();
    std::cout << "acts[1]="; pt.output[1].print_acts();
    std::cout << "acts[2]="; pt.output[2].print_acts();
    std::cout << std::endl;

    return 0;
}
