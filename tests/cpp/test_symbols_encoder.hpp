#ifndef TEST_SYMBOLS_ENCODER_HPP
#define TEST_SYMBOLS_ENCODER_HPP

#include "symbols_encoder.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdint>

void test_symbols_encoder() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test SymbolsEncoder" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    // setup constants
    const uint32_t MAX_SYMBOLS = 4; // number of statelets
    const uint32_t NUM_S = 64; // number of statelets
    
    std::cout << "SymbolsEncoder Construction" << std::endl;
    std::cout << "---------------------------" << std::endl;
    SymbolsEncoder e(MAX_SYMBOLS, NUM_S);
    std::cout << "passed" << std::endl;
    std::cout << std::endl;

    std::cout << "SymbolsEncoder Initialize" << std::endl;
    std::cout << "-------------------------" << std::endl;
    e.initialize();
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    std::cout << "SymbolsEncoder Compute" << std::endl;
    std::cout << "----------------------" << std::endl;

    e.compute(0);
    std::cout << "value=0" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(1);
    std::cout << "value=1" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(2);
    std::cout << "value=2" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(3);
    std::cout << "value=3" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;

    e.compute(4);
    std::cout << "value=4" << std::endl;
    std::cout << "output="; e.get_output()[CURR].print_bits();
    std::cout << std::endl;
}

#endif