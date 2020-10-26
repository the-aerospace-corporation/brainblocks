#ifndef TEST_PAGE_HPP
#define TEST_PAGE_HPP

#include "page.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_page() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test Page" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    //utils_seed(0);

    uint32_t curr = 0;
    uint32_t prev = 1;

    std::cout << "Page Construction" << std::endl;
    std::cout << "-----------------" << std::endl;
    Page p0(2, 32);
    Page p1(2, 32);
    Page p2(2, 0);
    std::cout << "passed" << std::endl;
    std::cout << std::endl;

    std::cout << "Page Add Child" << std::endl;
    std::cout << "--------------" << std::endl;
    p2.add_child(p0);
    p2.add_child(p1);
    std::cout << "passed" << std::endl;
    std::cout << std::endl;

    std::cout << "Page Initialize" << std::endl;
    std::cout << "---------------" << std::endl;
    p0.initialize();
    p1.initialize();
    p2.initialize();
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << "p1 bits[curr]="; p1.print_bits(curr);
    std::cout << "p1 acts[curr]="; p1.print_acts(curr);
    std::cout << "p1 bits[prev]="; p1.print_bits(prev);
    std::cout << "p1 acts[prev]="; p1.print_acts(prev);
    std::cout << "p2 bits[curr]="; p2.print_bits(curr);
    std::cout << "p2 acts[curr]="; p2.print_acts(curr);
    std::cout << "p2 bits[prev]="; p2.print_bits(prev);
    std::cout << "p2 acts[prev]="; p2.print_acts(prev);
    std::cout << std::endl;

    std::cout << "Page Set Bit" << std::endl;
    std::cout << "------------" << std::endl;
    p0.set_bit(curr, 0);
    p0.set_bit(prev, 1);
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << std::endl;

    std::cout << "Page Get Bit" << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << "bitarrays[curr][0]=" << p0.get_bit(curr, 0) << std::endl;
    std::cout << "bitarrays[prev][1]=" << p0.get_bit(prev, 1) << std::endl;
    std::cout << std::endl;

    std::cout << "Page Clear Bit" << std::endl;
    std::cout << "--------------" << std::endl;
    p0.clear_bit(curr, 0);
    p0.clear_bit(prev, 1);
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << std::endl;

    std::cout << "Page Set Acts" << std::endl;
    std::cout << "-------------" << std::endl;
    std::vector<uint32_t> acts0 = {0, 2, 3, 4, 5, 6, 7, 31};
    std::vector<uint32_t> acts1 = {0, 16, 18, 20, 22, 24, 26, 31};
    p0.set_acts(curr, acts0);
    p1.set_acts(curr, acts1);
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << "p1 bits[curr]="; p1.print_bits(curr);
    std::cout << "p1 acts[curr]="; p1.print_acts(curr);
    std::cout << "p1 bits[prev]="; p1.print_bits(prev);
    std::cout << "p1 acts[prev]="; p1.print_acts(prev);
    std::cout << "p2 bits[curr]="; p2.print_bits(curr);
    std::cout << "p2 acts[curr]="; p2.print_acts(curr);
    std::cout << "p2 bits[prev]="; p2.print_bits(prev);
    std::cout << "p2 acts[prev]="; p2.print_acts(prev);
    std::cout << std::endl;

    std::cout << "Page Fetch" << std::endl;
    std::cout << "----------" << std::endl;
    p2.fetch();
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << "p1 bits[curr]="; p1.print_bits(curr);
    std::cout << "p1 acts[curr]="; p1.print_acts(curr);
    std::cout << "p1 bits[prev]="; p1.print_bits(prev);
    std::cout << "p1 acts[prev]="; p1.print_acts(prev);
    std::cout << "p2 bits[curr]="; p2.print_bits(curr);
    std::cout << "p2 acts[curr]="; p2.print_acts(curr);
    std::cout << "p2 bits[prev]="; p2.print_bits(prev);
    std::cout << "p2 acts[prev]="; p2.print_acts(prev);
    std::cout << std::endl;

    std::cout << "Page Step" << std::endl;
    std::cout << "---------" << std::endl;
    p0.step();
    p1.step();
    p2.step();
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << "p1 bits[curr]="; p1.print_bits(curr);
    std::cout << "p1 acts[curr]="; p1.print_acts(curr);
    std::cout << "p1 bits[prev]="; p1.print_bits(prev);
    std::cout << "p1 acts[prev]="; p1.print_acts(prev);
    std::cout << "p2 bits[curr]="; p2.print_bits(curr);
    std::cout << "p2 acts[curr]="; p2.print_acts(curr);
    std::cout << "p2 bits[prev]="; p2.print_bits(prev);
    std::cout << "p2 acts[prev]="; p2.print_acts(prev);
    std::cout << std::endl;

    std::cout << "Page Compute Changed" << std::endl;
    std::cout << "--------------------" << std::endl;
    p0.compute_changed();
    p1.compute_changed();
    p2.compute_changed();
    std::cout << "p0_changed_flag=" << p0.has_changed() << std::endl;
    std::cout << "p1_changed_flag=" << p1.has_changed() << std::endl;
    std::cout << "p2_changed_flag=" << p2.has_changed() << std::endl;
    std::cout << std::endl;

    std::cout << "Page Copy Previous to Current" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    p0.copy_previous_to_current();
    p1.copy_previous_to_current();
    p2.copy_previous_to_current();
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << "p1 bits[curr]="; p1.print_bits(curr);
    std::cout << "p1 acts[curr]="; p1.print_acts(curr);
    std::cout << "p1 bits[prev]="; p1.print_bits(prev);
    std::cout << "p1 acts[prev]="; p1.print_acts(prev);
    std::cout << "p2 bits[curr]="; p2.print_bits(curr);
    std::cout << "p2 acts[curr]="; p2.print_acts(curr);
    std::cout << "p2 bits[prev]="; p2.print_bits(prev);
    std::cout << "p2 acts[prev]="; p2.print_acts(prev);
    std::cout << std::endl;

    std::cout << "Page Compute Changed (again)" << std::endl;
    std::cout << "----------------------------" << std::endl;
    p0.compute_changed();
    p1.compute_changed();
    p2.compute_changed();
    std::cout << "p0_changed_flag=" << p0.has_changed() << std::endl;
    std::cout << "p1_changed_flag=" << p1.has_changed() << std::endl;
    std::cout << "p2_changed_flag=" << p2.has_changed() << std::endl;
    std::cout << std::endl;

    std::cout << "Page Clear Bits" << std::endl;
    std::cout << "---------------" << std::endl;
    p0.clear_bits(curr);
    p0.clear_bits(prev);
    p1.clear_bits(curr);
    p1.clear_bits(prev);
    p2.clear_bits(curr);
    p2.clear_bits(prev);
    std::cout << "p0 bits[curr]="; p0.print_bits(curr);
    std::cout << "p0 acts[curr]="; p0.print_acts(curr);
    std::cout << "p0 bits[prev]="; p0.print_bits(prev);
    std::cout << "p0 acts[prev]="; p0.print_acts(prev);
    std::cout << "p1 bits[curr]="; p1.print_bits(curr);
    std::cout << "p1 acts[curr]="; p1.print_acts(curr);
    std::cout << "p1 bits[prev]="; p1.print_bits(prev);
    std::cout << "p1 acts[prev]="; p1.print_acts(prev);
    std::cout << "p2 bits[curr]="; p2.print_bits(curr);
    std::cout << "p2 acts[curr]="; p2.print_acts(curr);
    std::cout << "p2 bits[prev]="; p2.print_bits(prev);
    std::cout << "p2 acts[prev]="; p2.print_acts(prev);
    std::cout << std::endl;
}

#endif