// =============================================================================
// test_bitarray.cpp
// =============================================================================
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

    unsigned long idx = 0;
    word_t word = 2;

    std::cout << "trailing_zeros(word);" << std::endl;
    std::cout << "---------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    idx = trailing_zeros(word);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "word=" << word << std::endl;
    std::cout << " idx=" << idx << std::endl;
    std::cout << std::endl;

    std::cout << "leading_zeros(word);" << std::endl;
    std::cout << "---------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    idx = leading_zeros(word);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "word=" << word << std::endl;
    std::cout << " idx=" << idx << std::endl;
    std::cout << std::endl;

    const uint32_t NUM_BITS = 1024;

    std::cout << "BitArray ba(NUM_BITS);" << std::endl;
    std::cout << "----------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    BitArray ba(NUM_BITS);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "size=" << ba.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "ba.erase();" << std::endl;
    std::cout << "-----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.erase();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "size=" << ba.num_bits() << "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "ba.resize(NUM_BITS);" << std::endl;
    std::cout << "--------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.resize(NUM_BITS);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "size=" << ba.num_bits() <<  "bits" << std::endl;
    std::cout << std::endl;

    std::cout << "ba.set_bit(4);" << std::endl;
    std::cout << "--------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.set_bit(4);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.get_bit(4);" << std::endl;
    std::cout << "--------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    uint32_t val = ba.get_bit(4);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "ba[4]=" << val << std::endl;
    std::cout << std::endl;

    std::cout << "ba.clear_bit(4);" << std::endl;
    std::cout << "----------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.clear_bit(4);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.toggle_bit(7);" << std::endl;
    std::cout << "-----------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.toggle_bit(7);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.assign_bit(7, 0);" << std::endl;
    std::cout << "--------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.assign_bit(7, 0);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.set_range(2, 8);" << std::endl;
    std::cout << "-------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.set_range(2, 8);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.toggle_range(4, 8);" << std::endl;
    std::cout << "----------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.toggle_range(4, 8);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.clear_range(2, 10);" << std::endl;
    std::cout << "----------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.clear_range(2, 10);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.set_all();" << std::endl;
    std::cout << "-------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.set_all();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "bits="; ba.print_bits();
    std::cout << std::endl;

    std::cout << "ba.clear_all();" << std::endl;
    std::cout << "---------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.clear_all();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "bits="; ba.print_bits();
    std::cout << std::endl;

    std::cout << "ba.toggle_all();" << std::endl;
    std::cout << "----------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.toggle_all();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "bits="; ba.print_bits();
    std::cout << std::endl;

    std::cout << "ba.set_bits({0, 1, 0, 1, 0, 1, 0, 1});" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    ba.clear_all();
    std::vector<uint8_t> in_bits = {0, 1, 0, 1, 0, 1, 0, 1};
    t0 = std::chrono::high_resolution_clock::now();
    ba.set_bits(in_bits);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.set_acts({2, 4, 6, 8});" << std::endl;
    std::cout << "--------------------------" << std::endl;
    ba.clear_all();
    std::vector<uint32_t> in_acts = {2, 4, 6, 8};
    t0 = std::chrono::high_resolution_clock::now();
    ba.set_acts(in_acts);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.get_bits();" << std::endl;
    std::cout << "--------------" << std::endl;
    ba.clear_all();
    ba.set_range(4, 8);
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> out_bits = ba.get_bits();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "out_bits={";
    for (uint32_t i = 0; i < out_bits.size(); i++) {
        std::cout << (uint32_t)out_bits[i];
        if (i < out_bits.size() - 1)
            std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << std::endl;

    std::cout << "ba.get_acts();" << std::endl;
    std::cout << "--------------" << std::endl;
    ba.clear_all();
    ba.set_range(4, 8);
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> out_acts = ba.get_acts();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "out_acts={";
    for (uint32_t i = 0; i < out_acts.size(); i++) {
        std::cout << (uint32_t)out_acts[i];
        if (i < out_acts.size() - 1)
            std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << std::endl;

    std::cout << "ba.num_set();" << std::endl;
    std::cout << "-------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    uint32_t num_set = ba.num_set();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "num_set=" << num_set << std::endl;
    std::cout << std::endl;

    std::cout << "ba.num_cleared();" << std::endl;
    std::cout << "-----------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    uint32_t num_cleared = ba.num_cleared();
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << "num_cleared=" << num_cleared << std::endl;
    std::cout << std::endl;

    BitArray ba0(1024);
    BitArray ba1(1024);
    BitArray ba2(1024);

    std::cout << "ba2.num_similar(ba0);" << std::endl;
    std::cout << "---------------------" << std::endl;
    ba0.clear_all();
    ba2.clear_all();
    ba0.set_range(4, 8);
    ba2.set_range(6, 10);
    t0 = std::chrono::high_resolution_clock::now();
    uint32_t num_similar = ba2.num_similar(ba0);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba0 acts="; ba0.print_acts();
    std::cout << "ba2 acts="; ba2.print_acts();
    std::cout << "num_similar=" << num_similar << std::endl;
    std::cout << std::endl;

    bool success;
    uint32_t next_bit;

    std::cout << "ba.find_next_set_bit(0, &next_bit);" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    ba.clear_all();
    ba.set_range(4, 8);
    success = false;
    next_bit = 0xFFFFFFFF;
    t0 = std::chrono::high_resolution_clock::now();
    success = ba.find_next_set_bit(0, &next_bit);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba acts="; ba.print_acts();
    std::cout << "success=" << success << std::endl;
    std::cout << "next_bit=" << next_bit << std::endl;
    std::cout << std::endl;

    std::cout << "ba.find_next_set_bit(6, 18, &next_bit);" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    ba.clear_all();
    ba.set_range(4, 8);
    success = false;
    next_bit = 0xFFFFFFFF;
    t0 = std::chrono::high_resolution_clock::now();
    success = ba.find_next_set_bit(6, 18, &next_bit);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba acts="; ba.print_acts();
    std::cout << "success=" << success << std::endl;
    std::cout << "next_bit=" << next_bit << std::endl;
    std::cout << std::endl;

    std::mt19937 rng(0);

    std::cout << "ba.random_shuffle();" << std::endl;
    std::cout << "--------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.random_shuffle(rng);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.random_set_num(100);" << std::endl;
    std::cout << "-----------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.random_set_num(rng, 100);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.random_set_pct(0.1);" << std::endl;
    std::cout << "-----------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.random_set_pct(rng, 0.1);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    std::cout << "ba.random_set_pct(0.1);" << std::endl;
    std::cout << "-----------------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba.random_set_pct(rng, 0.1);
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "acts="; ba.print_acts();
    std::cout << std::endl;

    ba0.clear_all();
    ba1.clear_all();
    ba2.clear_all();
    ba0.set_bit(2);
    ba0.set_bit(3);
    ba1.set_bit(1);
    ba1.set_bit(3);

    std::cout << "ba2 = ~ba0" << std::endl;
    std::cout << "----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba2 = ~ba0;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba0 acts="; ba0.print_acts();
    std::cout << "ba2 bits="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "ba2 = ba0 & ba1" << std::endl;
    std::cout << "---------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba2 = ba0 & ba1;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba0 acts="; ba0.print_acts();
    std::cout << "ba1 acts="; ba1.print_acts();
    std::cout << "ba2 bits="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "ba2 = ba0 | ba1" << std::endl;
    std::cout << "---------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba2 = ba0 | ba1;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba0 acts="; ba0.print_acts();
    std::cout << "ba1 acts="; ba1.print_acts();
    std::cout << "ba2 bits="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "ba2 = ba0 ^ ba1" << std::endl;
    std::cout << "---------------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    ba2 = ba0 ^ ba1;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba0 acts="; ba0.print_acts();
    std::cout << "ba1 acts="; ba1.print_acts();
    std::cout << "ba2 bits="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "ba0 == ba1" << std::endl;
    std::cout << "----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    uint32_t is_equal = ba0 == ba1;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba0 acts="; ba0.print_acts();
    std::cout << "ba1 acts="; ba1.print_acts();
    std::cout << "is_equal=" << is_equal << std::endl;
    std::cout << std::endl;

    std::cout << "ba0 != ba1" << std::endl;
    std::cout << "----------" << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    uint32_t not_equal = ba0 != ba1;
    t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
    std::cout << "t=" << duration.count() << "s" << std::endl;
    std::cout << "ba0 acts="; ba0.print_acts();
    std::cout << "ba1 acts="; ba1.print_acts();
    std::cout << "not_equal=" << not_equal << std::endl;
    std::cout << std::endl;

    return 0;
}
