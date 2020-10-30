#ifndef TEST_SCALAR_ENCODER_HPP
#define TEST_SCALAR_ENCODER_HPP

#include "scalar_encoder.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdint>

void test_scalar_encoder() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test ScalarEncoder" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    // setup constants
    const double MIN_VAL = -1.0; // minimum input value
    const double MAX_VAL = 1.0;  // maximum input value
    const uint32_t NUM_S = 64;   // number of statelets
    const uint32_t NUM_AS = 8;   // number of active statelets
    
    std::cout << "ScalarEncoder Construction" << std::endl;
    std::cout << "--------------------------" << std::endl;
    ScalarEncoder e(MIN_VAL, MAX_VAL, NUM_S, NUM_AS);
    std::cout << "passed" << std::endl;
    std::cout << std::endl;

    std::cout << "ScalarEncoder Initialize" << std::endl;
    std::cout << "------------------------" << std::endl;
    e.initialize();
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    std::cout << "ScalarEncoder Compute" << std::endl;
    std::cout << "---------------------" << std::endl;

    e.compute(-1.5);
    std::cout << "value=-1.5" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(-1.0);
    std::cout << "value=-1.0" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(-0.5);
    std::cout << "value=-0.5" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.5);
    std::cout << "value=0.5" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(1.0);
    std::cout << "value=1.0" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(1.5);
    std::cout << "value=1.5" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;
}

#endif