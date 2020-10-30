#ifndef TEST_SEQUENCE_LEARNER_HPP
#define TEST_SEQUENCE_LEARNER_HPP

#include "scalar_encoder.hpp"
#include "sequence_learner.hpp"
#include "utils.hpp"
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>

void test_sequence_learner() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test SequenceLearner" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    // setup constants
    const double MIN_VAL = 0.0;  // e minimum input value
    const double MAX_VAL = 1.0;  // e maximum input value
    const uint32_t NUM_S = 64;   // e number of statelets
    const uint32_t NUM_AS = 8;   // e number of active statelets
    const uint32_t NUM_SPC = 10; // sl number of statelets per column
    const uint32_t NUM_DPS = 10; // sl number of coincidence detectors per statelet
    const uint32_t NUM_RPD = 12; // sl number of receptors per coincidence detector
    const uint32_t D_THRESH = 6; // sl coincidence detector threshold
    const uint8_t PERM_THR = 1;  // sl receptor permanence threshold
    const uint8_t PERM_INC = 1;  // sl receptor permanence increment
    const uint8_t PERM_DEC = 0;  // sl receptor permanence decrement

    // setup data
    double values[30] = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    double scores[30];

    for (uint32_t i = 0; i < 30; i++) {
        scores[i] = 0.0;
    }

    std::cout << "SequenceLearner Construction" << std::endl;
    std::cout << "----------------------------" << std::endl;
    ScalarEncoder e(MIN_VAL, MAX_VAL, NUM_S, NUM_AS);
    SequenceLearner sl(NUM_SPC, NUM_DPS, NUM_RPD, D_THRESH, PERM_THR, PERM_INC, PERM_DEC);
    sl.get_input().add_child(e.get_output());
    std::cout << "Complete" << std::endl;
    std::cout << std::endl;

    std::cout << "SequenceLearner Running Scenario" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    for (uint32_t i = 0; i < 30; i++) {
        e.compute(values[i]);
        sl.compute(true);
        scores[i] = sl.get_score();
    }
    std::cout << "Complete" << std::endl;
    std::cout << std::endl;

    std::cout << "values, scores" << std::endl;
    std::cout << std::fixed;
    for (uint32_t i = 0; i < 30; i++) {
        std::cout << std::setprecision(4) << values[i] << ", " << scores[i] << std::endl;
    }
    std::cout << std::endl;
}

#endif