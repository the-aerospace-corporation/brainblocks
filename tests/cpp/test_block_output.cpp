// =============================================================================
// test_block_output.cpp
// =============================================================================
#include "block_output.hpp"
#include <iostream>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    const uint32_t NUM_BITS = 1024;

    std::cout << "BlockOutput out;" << std::endl;
    std::cout << "----------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    BlockOutput out;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "out.setup(3, NUM_BITS);" << std::endl;
    std::cout << "-----------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.setup(3, NUM_BITS);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.step();" << std::endl;
    std::cout << "-----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.step();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.state.set_range(0, 8);" << std::endl;
    std::cout << "--------------------------" << std::endl;
    out.state.clear_all();
    t0 = std::chrono::high_resolution_clock::now();
    out.state.set_range(0, 8);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.store();" << std::endl;
    std::cout << "------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.store();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.step();" << std::endl;
    std::cout << "-----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.step();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.state.set_range(8, 8);" << std::endl;
    std::cout << "--------------------------" << std::endl;
    out.state.clear_all();
    t0 = std::chrono::high_resolution_clock::now();
    out.state.set_range(8, 8);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.store();" << std::endl;
    std::cout << "------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.store();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.step();" << std::endl;
    std::cout << "-----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.step();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.state.set_range(16, 8);" << std::endl;
    std::cout << "--------------------------" << std::endl;
    out.state.clear_all();
    t0 = std::chrono::high_resolution_clock::now();
    out.state.set_range(16, 8);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.store();" << std::endl;
    std::cout << "------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.store();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.step();" << std::endl;
    std::cout << "-----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.step();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.state.set_range(24, 8);" << std::endl;
    std::cout << "---------------------------" << std::endl;
    out.state.clear_all();
    t0 = std::chrono::high_resolution_clock::now();
    out.state.set_range(24, 8);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    std::cout << "out.store();" << std::endl;
    std::cout << "------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    out.store();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "  state acts="; out.state.print_acts();
    std::cout << "hist acts[0]="; out[0].print_acts();
    std::cout << "hist acts[1]="; out[1].print_acts();
    std::cout << "hist acts[2]="; out[2].print_acts();
    std::cout << std::endl;

    return 0;
}
