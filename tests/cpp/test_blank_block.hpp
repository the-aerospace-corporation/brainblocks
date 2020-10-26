#ifndef TEST_BLANK_BLOCK_HPP
#define TEST_BLANK_BLOCK_HPP

#include "blank_block.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_blank_block() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test Blank Block" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;
    
    //utils_seed(0);

    uint32_t curr = 0;
    uint32_t prev = 1;

    std::cout << "Blank Block Construction" << std::endl;
    std::cout << "------------------------" << std::endl;
    BlankBlock bb(32);
    std::cout << "bb output="; bb.get_output().print_bits(curr);
    std::cout << std::endl;

    bb.print_parameters();
}

#endif