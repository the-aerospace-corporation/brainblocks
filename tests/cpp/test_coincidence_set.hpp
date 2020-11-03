#ifndef TEST_COINCIDENCE_SET_HPP
#define TEST_COINCIDENCE_SET_HPP

#include "coincidence_set.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_coincidence_set() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test CoincidenceSet" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    // setup constants
    const uint32_t NUM_I = 64;   // number of inputs
    const uint32_t NUM_AI = 32;  // number of active inputs
    const uint32_t NUM_R = 16;   // number of receptors
    const uint32_t NUM_CONN = 8; // number of receptors initially connected
    const uint32_t PERM_THR = 4; // permanence threshold
    const uint32_t PERM_INC = 2; // permanence increment
    const uint32_t PERM_DEC = 1; // permanence decrement

    // setup input bitarray
    BitArray input_ba(NUM_I);

    for (uint32_t i = 0; i < NUM_AI; i++) {
        input_ba.set_bit(i, 1);
    }

    // setup learn mask bitarray
    BitArray lmask_ba(NUM_R);
    lmask_ba.random_fill(0.8);

    std::cout << "CoincidenceSet Construction" << std::endl;
    std::cout << "---------------------------" << std::endl;
    CoincidenceSet cs;
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << std::endl;

    std::cout << "CoincidenceSet Resize" << std::endl;
    std::cout << "---------------------" << std::endl;
    cs.resize(NUM_R);
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << std::endl;

    std::cout << "CoincidenceSet Initialize Pool" << std::endl;
    std::cout << "------------------------------" << std::endl;
    cs.initialize_pool(NUM_R, NUM_I, NUM_CONN, PERM_THR);
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << std::endl;

    std::cout << "CoincidenceSet Overlap" << std::endl;
    std::cout << "----------------------" << std::endl;
    uint32_t overlap = cs.overlap(input_ba, PERM_THR);
    std::cout << "input="; input_ba.print_bits();
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << "overlap=" << overlap << std::endl;
    std::cout << std::endl;

    lmask_ba.random_shuffle();

    std::cout << "CoincidenceSet Learn" << std::endl;
    std::cout << "--------------------" << std::endl;
    cs.learn(input_ba, lmask_ba, PERM_INC, PERM_DEC);
    std::cout << "input="; input_ba.print_bits();
    std::cout << "lmask="; lmask_ba.print_bits();
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << std::endl;

    std::cout << "CoincidenceSet Punish" << std::endl;
    std::cout << "---------------------" << std::endl;
    cs.punish(input_ba, lmask_ba, PERM_INC);
    std::cout << "input="; input_ba.print_bits();
    std::cout << "lmask="; lmask_ba.print_bits();
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << std::endl;

    std::cout << "CoincidenceSet Learn (Perm Bounds)" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    cs.learn(input_ba, lmask_ba, 100, 100);
    std::cout << "input="; input_ba.print_bits();
    std::cout << "lmask="; lmask_ba.print_bits();
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << std::endl;

    std::cout << "CoincidenceSet Learn (Move)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    cs.learn_move(input_ba, lmask_ba, PERM_INC, PERM_DEC, PERM_THR);
    std::cout << "input="; input_ba.print_bits();
    std::cout << "lmask="; lmask_ba.print_bits();
    std::cout << "addrs="; cs.print_addrs();
    std::cout << "perms="; cs.print_perms();
    std::cout << std::endl;
}

#endif