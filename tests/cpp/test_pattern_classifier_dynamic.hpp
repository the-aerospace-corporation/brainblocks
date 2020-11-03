#ifndef TEST_PATTERN_CLASSIFIER_DYNAMIC_HPP
#define TEST_PATTERN_CLASSIFIER_DYNAMIC_HPP

#include "pattern_classifier_dynamic.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_pattern_classifier_dynamic() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test PatternClassifierDynamic" << std::endl;
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
    const uint32_t NUM_SPL = 8;   // pc number of statelets per label
    const uint8_t PERM_THR = 20;  // pc receptor permanence threshold
    const uint8_t PERM_INC = 2;   // pc receptor permanence increment
    const uint8_t PERM_DEC = 1;   // pc receptor permanence decrement
    const double PCT_POOL = 0.8;  // pc pooling percentage
    const double PCT_CONN = 0.5;  // pc initially connected percentage
    const double PCT_LEARN = 0.3; // pc learn percentage

    std::vector<uint32_t> set_labels = {0, 1};

    std::cout << "PatternClassifierDynamic Construction" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    ScalarEncoder e(MIN_VAL, MAX_VAL, NUM_I, NUM_AI);
    PatternClassifierDynamic pc(NUM_S, NUM_AS, NUM_SPL, PERM_THR, PERM_INC, PERM_DEC, PCT_POOL, PCT_CONN, PCT_LEARN);
    pc.get_input().add_child(e.get_output());
    std::cout << "Complete" << std::endl;
    std::cout << std::endl;


    std::cout << "PatternClassifierDynamic Running Scenario" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    std::vector<uint32_t> labels;
    std::vector<double> probs;

    std::cout << "Before Learning:" << std::endl;
    std::cout << std::endl;

    e.compute(0.0);
    pc.compute(0, false);
    labels = pc.get_labels();
    probs = pc.get_probabilities();
    std::cout << "input_value=0.0" << std::endl;
    std::cout << "input_label=0" << std::endl;
    std::cout << "learn_flag=false" << std::endl;
    std::cout << "pc input ="; pc.get_input()[CURR].print_bits();
    std::cout << "pc output="; pc.get_output()[CURR].print_bits();
    for (uint32_t l = 0; l < labels.size(); l++) {
        std::cout << "P(label=" << labels[l] << ")=" << probs[l] << std::endl;
    }
    std::cout << std::endl;

    e.compute(1.0);
    pc.compute(0, false);
    labels = pc.get_labels();
    probs = pc.get_probabilities();
    std::cout << "input_value=1.0" << std::endl;
    std::cout << "input_label=1" << std::endl;
    std::cout << "learn_flag=false" << std::endl;
    std::cout << "pc input ="; pc.get_input()[CURR].print_bits();
    std::cout << "pc output="; pc.get_output()[CURR].print_bits();
    for (uint32_t l = 0; l < labels.size(); l++) {
        std::cout << "P(label=" << labels[l] << ")=" << probs[l] << std::endl;
    }
    std::cout << std::endl;

    for (uint32_t i = 0; i < 5; i++) {
        e.compute(0.0);
        pc.compute(0, true);
        e.compute(1.0);
        pc.compute(1, true);
    }
    std::cout << "After Learning:" << std::endl;
    std::cout << std::endl;

    e.compute(0.0);
    pc.compute();
    labels = pc.get_labels();
    probs = pc.get_probabilities();
    std::cout << "input_value=0.0" << std::endl;
    std::cout << "input_label=N/A" << std::endl;
    std::cout << "learn_flag=false" << std::endl;
    std::cout << "pc input ="; pc.get_input()[CURR].print_bits();
    std::cout << "pc output="; pc.get_output()[CURR].print_bits();
    for (uint32_t l = 0; l < labels.size(); l++) {
        std::cout << "P(label=" << labels[l] << ")=" << probs[l] << std::endl;
    }
    std::cout << std::endl;

    e.compute(1.0);
    pc.compute();
    labels = pc.get_labels();
    probs = pc.get_probabilities();
    std::cout << "input_value=1.0" << std::endl;
    std::cout << "input_label=N/A" << std::endl;
    std::cout << "learn_flag=false" << std::endl;
    std::cout << "pc input ="; pc.get_input()[CURR].print_bits();
    std::cout << "pc output="; pc.get_output()[CURR].print_bits();
    for (uint32_t l = 0; l < labels.size(); l++) {
        std::cout << "P(label=" << labels[l] << ")=" << probs[l] << std::endl;
    }
    std::cout << std::endl;

    e.compute(0.0);
    pc.compute();
    labels = pc.get_labels();
    probs = pc.get_probabilities();
    std::cout << "input_value=0.0" << std::endl;
    std::cout << "input_label=N/A" << std::endl;
    std::cout << "learn_flag=false" << std::endl;
    std::cout << "pc input ="; pc.get_input()[CURR].print_bits();
    std::cout << "pc output="; pc.get_output()[CURR].print_bits();
    for (uint32_t l = 0; l < labels.size(); l++) {
        std::cout << "P(label=" << labels[l] << ")=" << probs[l] << std::endl;
    }
    std::cout << std::endl;

    e.compute(1.0);
    pc.compute();
    labels = pc.get_labels();
    probs = pc.get_probabilities();
    std::cout << "input_value=1.0" << std::endl;
    std::cout << "input_label=N/A" << std::endl;
    std::cout << "learn_flag=false" << std::endl;
    std::cout << "pc input ="; pc.get_input()[CURR].print_bits();
    std::cout << "pc output="; pc.get_output()[CURR].print_bits();
    for (uint32_t l = 0; l < labels.size(); l++) {
        std::cout << "P(label=" << labels[l] << ")=" << probs[l] << std::endl;
    }
    std::cout << std::endl;
}

#endif