#ifndef TEST_PERSISTENCE_ENCODER_HPP
#define TEST_PERSISTENCE_ENCODER_HPP

#include "persistence_encoder.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_persistence_encoder() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test PersistenceEncoder" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    // setup constants
    const double MIN_VAL = -1.0;  // minimum input value
    const double MAX_VAL = 1.0;   // maximum input value
    const uint32_t NUM_S = 64;    // number of statelets
    const uint32_t NUM_AS = 8;    // number of active statelets
    const uint32_t MAX_STEPS = 4; // max steps
    
    std::cout << "PersistenceEncoder Construction" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    PersistenceEncoder e(MIN_VAL, MAX_VAL, NUM_S, NUM_AS, MAX_STEPS);
    std::cout << "passed" << std::endl;
    std::cout << std::endl;

    std::cout << "PersistenceEncoder Initialize" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    e.initialize();
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;

    std::cout << "PersistenceEncoder Compute" << std::endl;
    std::cout << "--------------------------" << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;

    e.compute(0.0);
    std::cout << "value=0.0" << std::endl;
    std::cout << "output="; e.output[CURR].print_bits();
    std::cout << std::endl;
}

#endif