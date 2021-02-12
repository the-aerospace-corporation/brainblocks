// =============================================================================
// block_input.cpp
// =============================================================================
#include "block_input.hpp"
#include "block_output.hpp"
#include <iostream>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    const uint32_t NUM_BITS = 1024;

    BlockOutput out0;
    BlockOutput out1;
    out0.setup(3, NUM_BITS);
    out1.setup(3, NUM_BITS);

    std::cout << "BlockInput in;" << std::endl;
    std::cout << "--------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    BlockInput in;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "in.add_child(&out0, PREV);" << std::endl;
    std::cout << "--------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    in.add_child(&out0, PREV);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "in size=" << in.state.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "in.add_child(&out0, CURR);" << std::endl;
    std::cout << "--------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    in.add_child(&out1, CURR);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "in size=" << in.state.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    // Simulate a time step
    out0.step();
    out1.step();
    out0.state.clear_all();
    out1.state.clear_all();
    out0.state.set_range(0, 8);
    out1.state.set_range(1016, 8);
    out0.store();
    out1.store();

    // Simulate a time step
    out0.step();
    out1.step();
    out0.state.clear_all();
    out1.state.clear_all();
    out0.state.set_range(8, 8);
    out1.state.set_range(1008, 8);
    out0.store();
    out1.store();

    // Simulate a time step
    out0.step();
    out1.step();
    out0.state.clear_all();
    out1.state.clear_all();
    out0.state.set_range(16, 8);
    out1.state.set_range(1000, 8);
    out0.store();
    out1.store();

    std::cout << "in.pull();" << std::endl;
    std::cout << "----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    in.pull();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "out0.state="; out0.state.print_acts();
    std::cout << "out0[0]   ="; out0[0].print_acts();
    std::cout << "out0[1]   ="; out0[1].print_acts();
    std::cout << "out0[2]   ="; out0[2].print_acts();
    std::cout << "out1.state="; out1.state.print_acts();
    std::cout << "out1[0]   ="; out1[0].print_acts();
    std::cout << "out1[1]   ="; out1[1].print_acts();
    std::cout << "out1[2]   ="; out1[2].print_acts();
    std::cout << "  in.state="; in.state.print_acts();
    std::cout << std::endl;

    std::cout << "in.push();" << std::endl;
    std::cout << "----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    in.pull();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  in.state="; in.state.print_acts();
    std::cout << "out0.state="; out0.state.print_acts();
    std::cout << "out1.state="; out1.state.print_acts();
    std::cout << std::endl;

    std::cout << "in.clear();" << std::endl;
    std::cout << "-----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    in.clear();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  in.state="; in.state.print_acts();
    std::cout << std::endl;

    return 0;
}
