// =============================================================================
// test_block_memory.cpp
// =============================================================================
#include "block_memory.hpp"
#include "bitarray.hpp"
#include <iostream>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    std::mt19937 rng(0);
    uint32_t NUM_BITS = 1024;

    BitArray input(NUM_BITS);
    input.set_range(0, 128);

    std::cout << "BlockMemory mem;" << std::endl;
    std::cout << "----------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    BlockMemory mem;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "mem.init_pooled_conn(...)" << std::endl;
    std::cout << "-------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    mem.init_pooled_conn(NUM_BITS, 1, 0.8, 0.5, 0.3, 20, 2, 1, rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "memory_usage=" << mem.memory_usage() << " bytes" << std::endl;
    std::cout << std::endl;

    uint32_t overlap = 0xFFFFFFFF;

    std::cout << "mem.overlap(d, input)" << std::endl;
    std::cout << "--------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    overlap = mem.overlap(0, input);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "overlap=" << overlap << std::endl;
    std::cout << std::endl;

    std::cout << "mem.overlap_conn(d, input)" << std::endl;
    std::cout << "--------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    overlap = mem.overlap_conn(0, input);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "overlap=" << overlap << std::endl;
    std::cout << std::endl;

    std::cout << "mem.learn(d, input)" << std::endl;
    std::cout << "------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    mem.learn(0, input, rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "mem.learn_conn(d, input)" << std::endl;
    std::cout << "------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    mem.learn_conn(0, input, rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "mem.learn_move(d, input)" << std::endl;
    std::cout << "------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    mem.learn_move(0, input, rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "mem.learn_move_conn(d, input)" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    mem.learn_move_conn(0, input, rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "mem.punish(d, input)" << std::endl;
    std::cout << "--------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    mem.punish(0, input, rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "mem.punish_conn(d, input)" << std::endl;
    std::cout << "-------------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    mem.punish_conn(0, input, rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "200 steps of learning" << std::endl;
    std::cout << "---------------------" << std::endl;

    for (uint32_t i = 0; i < 200; i++)
        mem.learn_move_conn(0, input, rng);

    std::cout << " input="; input.print_bits();
    std::cout << "d_conn="; mem.print_conns(0);
    std::cout << " addrs="; mem.print_addrs(0);
    std::cout << " perms="; mem.print_perms(0);

    return 0;
}
