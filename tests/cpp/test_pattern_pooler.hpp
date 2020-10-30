#ifndef TEST_PATTERN_POOLER_HPP
#define TEST_PATTERN_POOLER_HPP

#include "pattern_pooler.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_pattern_pooler() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test PatternPooler" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    // setup constants
    const double MIN_VAL = 0.0;   // e minimum input value
    const double MAX_VAL = 1.0;   // e maximum input value
    const uint32_t NUM_I = 64;    // e number of statelets
    const uint32_t NUM_AI = 32;   // e number of active statelets
    const uint32_t NUM_S = 64;    // pc number of statelets
    const uint32_t NUM_AS = 5;    // pc number of active statelets
    const uint8_t PERM_THR = 20;  // pp receptor permanence threshold
    const uint8_t PERM_INC = 2;   // pp receptor permanence increment
    const uint8_t PERM_DEC = 1;   // pp receptor permanence decrement
    const double PCT_POOL = 0.8;  // pp pooling percentage
    const double PCT_CONN = 0.5;  // pp initially connected percentage
    const double PCT_LEARN = 0.3; // pp learn percentage

    std::cout << "PatternPooler Construction" << std::endl;
    std::cout << "--------------------------" << std::endl;
    ScalarEncoder e(MIN_VAL, MAX_VAL, NUM_I, NUM_AI);
    PatternPooler pp(NUM_S, NUM_AS, PERM_THR, PERM_INC, PERM_DEC, PCT_POOL, PCT_CONN, PCT_LEARN);
    pp.get_input().add_child(e.get_output());
    std::cout << "Complete" << std::endl;
    std::cout << std::endl;

    std::cout << "PatternClassifier Running Scenario" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    std::cout << "Before Learning:" << std::endl;
    std::cout << std::endl;

    e.compute(0.0);
    pp.compute();
    std::cout << "input_value=0.0" << std::endl;
    std::cout << "pp input ="; pp.get_input()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_acts();
    std::cout << std::endl;

    e.compute(1.0);
    pp.compute();
    std::cout << "input_value=1.0" << std::endl;
    std::cout << "pp input ="; pp.get_input()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_acts();
    std::cout << std::endl;

    for (uint32_t i = 0; i < 5; i++) {
        e.compute(0.0);
        pp.compute(true);
        e.compute(1.0);
        pp.compute(true);
    }
    std::cout << "After Learning:" << std::endl;
    std::cout << std::endl;

    e.compute(0.0);
    pp.compute();
    std::cout << "input_value=0.0" << std::endl;
    std::cout << "pp input ="; pp.get_input()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_acts();
    std::cout << std::endl;

    e.compute(1.0);
    pp.compute();
    std::cout << "input_value=1.0" << std::endl;
    std::cout << "pp input ="; pp.get_input()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_bits();
    std::cout << "pp output="; pp.get_output()[CURR].print_acts();
    std::cout << std::endl;
}

#endif