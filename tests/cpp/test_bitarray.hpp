#ifndef TEST_BITARRAY_HPP
#define TEST_BITARRAY_HPP

#include "bitarray.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_bitarray() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test BitArray" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    std::cout << "BitArray Construction" << std::endl;
    std::cout << "---------------------" << std::endl;
    BitArray ba0(32);
    BitArray ba1(64);
    BitArray ba2(64);
    ba0.print_info();
    ba1.print_info();
    ba2.print_info();
    std::cout << std::endl;

    std::cout << "BitArray Resize" << std::endl;
    std::cout << "---------------" << std::endl;
    std::cout << "before="; ba0.print_bits();
    ba0.resize(64);
    std::cout << " after="; ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Random Fill" << std::endl;
    std::cout << "--------------------" << std::endl;
    ba0.random_fill(0.5);
    ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Random Shuffle" << std::endl;
    std::cout << "-----------------------" << std::endl;
    ba0.random_shuffle();
    ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Fill Bits" << std::endl;
    std::cout << "------------------" << std::endl;
    ba0.fill_bits();
    ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Clear Bits" << std::endl;
    std::cout << "-------------------" << std::endl;
    ba0.clear_bits();
    ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Set Bit to 1" << std::endl;
    std::cout << "---------------------" << std::endl;
    ba0.set_bit(4, 1);
    ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Get Bit" << std::endl;
    std::cout << "----------------" << std::endl;
    ba0.print_bits();
    std::cout << "bitarray[3]=" << (uint32_t)ba0.get_bit(3) << std::endl;
    std::cout << "bitarray[4]=" << (uint32_t)ba0.get_bit(4) << std::endl;
    std::cout << "bitarray[5]=" << (uint32_t)ba0.get_bit(5) << std::endl;
    std::cout << std::endl;

    std::cout << "BitArray Set Bit to 0" << std::endl;
    std::cout << "---------------------" << std::endl;
    ba0.set_bit(4, 0);
    ba0.print_bits();
    std::cout << std::endl;

    ba0.resize(32);
    ba0.clear_bits();

    std::cout << "BitArray Set Bits" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::vector<uint8_t> in_bits(ba0.get_num_bits(), 0);
    in_bits[0] = 1;
    in_bits[2] = 1;
    in_bits[4] = 1;
    in_bits[6] = 1;
    ba0.set_bits(in_bits);
    std::cout << "in_bits=[";
    for (uint32_t i = 0; i < in_bits.size(); i++) {
        std::cout << (uint32_t)in_bits[i];
        if (i < in_bits.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "bitarray="; ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Get Bits" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::vector<uint8_t> out_bits = ba0.get_bits();
    std::cout << "bitarray="; ba0.print_bits();
    std::cout << "out_bits=[";
    for (uint32_t i = 0; i < out_bits.size(); i++) {
        std::cout << (uint32_t)out_bits[i];
        if (i < out_bits.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    ba0.clear_bits();

    std::cout << "BitArray Set Acts" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::vector<uint32_t> in_acts(4);
    in_acts[0] = 0;
    in_acts[1] = 2;
    in_acts[2] = 4;
    in_acts[3] = 6;
    ba0.set_acts(in_acts);
    std::cout << "in_acts=[";
    for (uint32_t i = 0; i < in_acts.size(); i++) {
        std::cout << in_acts[i];
        if (i < in_acts.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "bitarray="; ba0.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Get Acts" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::vector<uint32_t> out_acts = ba0.get_acts();
    std::cout << "bitarray="; ba0.print_bits();
    std::cout << "out_acts=[";
    for (uint32_t i = 0; i < out_acts.size(); i++) {
        std::cout << out_acts[i];
        if (i < out_acts.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    ba0.clear_bits();
    ba0.resize(64);

    std::cout << "BitArray Copy" << std::endl;
    std::cout << "-------------" << std::endl;
    ba0.clear_bits();
    ba1.clear_bits();
    ba0.random_fill(0.5);
    std::cout << "before_src="; ba0.print_bits();
    std::cout << "before_dst="; ba1.print_bits();
    bitarray_copy(ba1, ba0, 0, 0, 2);
    std::cout << " after_src="; ba0.print_bits();
    std::cout << " after_dst="; ba1.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Copy (Offset and Subset)" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    ba0.clear_bits();
    ba1.clear_bits();
    ba0.random_fill(0.5);
    std::cout << "before_src="; ba0.print_bits();
    std::cout << "before_dst="; ba1.print_bits();
    bitarray_copy(ba1, ba0, 1, 0, 1);
    std::cout << " after_src="; ba0.print_bits();
    std::cout << " after_dst="; ba1.print_bits();
    std::cout << std::endl;

    ba0.clear_bits();
    ba1.clear_bits();
    ba0.random_fill(0.5);
    ba1.random_fill(0.5);

    std::cout << "BitArray Not" << std::endl;
    std::cout << "------------" << std::endl;
    ba2 = ~ba0;
    std::cout << " in="; ba0.print_bits();
    std::cout << "out="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray And" << std::endl;
    std::cout << "------------" << std::endl;
    ba2.clear_bits();
    ba2 = ba0 & ba1;
    std::cout << "in0="; ba0.print_bits();
    std::cout << "in1="; ba1.print_bits();
    std::cout << "out="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Or" << std::endl;
    std::cout << "-----------" << std::endl;
    ba2.clear_bits();
    ba2 = ba0 | ba1;
    std::cout << "in0="; ba0.print_bits();
    std::cout << "in1="; ba1.print_bits();
    std::cout << "out="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Xor" << std::endl;
    std::cout << "------------" << std::endl;
    ba2.clear_bits();
    ba2 = ba0 ^ ba1;
    std::cout << "in0="; ba0.print_bits();
    std::cout << "in1="; ba1.print_bits();
    std::cout << "out="; ba2.print_bits();
    std::cout << std::endl;

    std::cout << "BitArray Count" << std::endl;
    std::cout << "--------------" << std::endl;
    ba0.print_bits();
    std::cout << "count=" << ba0.count() << std::endl;
    std::cout << std::endl;
}

#endif