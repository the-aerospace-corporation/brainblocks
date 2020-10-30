#ifndef TEST_PAGE_HPP
#define TEST_PAGE_HPP

#include "page.hpp"
#include "utils.hpp"
#include <iostream>
#include <cstdint>
#include <vector>

void test_page() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test Page" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::endl;

    utils_seed(0);

    std::cout << "Page Construction" << std::endl;
    std::cout << "-----------------" << std::endl;
    Page p0;
    Page p1;
    Page p2;
    p0.set_num_bits(32);
    p1.set_num_bits(32);
    p0.print_info();
    p1.print_info();
    p2.print_info();
    std::cout << std::endl;

    std::cout << "Page Add Child" << std::endl;
    std::cout << "--------------" << std::endl;
    p2.add_child(p0);
    p2.add_child(p1);
    p2.print_info();
    std::cout << std::endl;

    std::cout << "Page Initialize" << std::endl;
    std::cout << "---------------" << std::endl;
    p0.initialize();
    p1.initialize();
    p2.initialize();
    p0.print_info();
    p1.print_info();
    p2.print_info();
    std::cout << std::endl;

    std::cout << "Page Set Bit to 1" << std::endl;
    std::cout << "-----------------" << std::endl;
    p0[CURR].set_bit(0, 1);
    p0[PREV].set_bit(1, 1);
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << std::endl;

    std::cout << "Page Get Bit" << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << "p0[CURR].get_bit(0)=" << p0[CURR].get_bit(0) << std::endl;
    std::cout << "p0[CURR].get_bit(1)=" << p0[CURR].get_bit(1) << std::endl;
    std::cout << "p0[PREV].get_bit(0)=" << p0[PREV].get_bit(0) << std::endl;
    std::cout << "p0[PREV].get_bit(1)=" << p0[PREV].get_bit(1) << std::endl;
    std::cout << std::endl;

    std::cout << "Page Set Bit to 0" << std::endl;
    std::cout << "-----------------" << std::endl;
    p0[CURR].set_bit(0, 0);
    p0[PREV].set_bit(1, 0);
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << std::endl;

    std::cout << "Page Set Acts" << std::endl;
    std::cout << "-------------" << std::endl;
    std::vector<uint32_t> acts0 = {0, 2, 3, 4, 5, 6, 7, 31};
    std::vector<uint32_t> acts1 = {0, 16, 18, 20, 22, 24, 26, 31};
    p0[CURR].set_acts(acts0);
    p1[CURR].set_acts(acts1);
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << "p1[CURR].bits="; p1[CURR].print_bits();
    std::cout << "p1[CURR].acts="; p1[CURR].print_acts();
    std::cout << "p1[PREV].bits="; p1[PREV].print_bits();
    std::cout << "p1[PREV].acts="; p1[PREV].print_acts();
    std::cout << "p2[CURR].bits="; p2[CURR].print_bits();
    std::cout << "p2[CURR].acts="; p2[CURR].print_acts();
    std::cout << "p2[PREV].bits="; p2[PREV].print_bits();
    std::cout << "p2[PREV].acts="; p2[PREV].print_acts();
    std::cout << std::endl;

    std::cout << "Page Fetch" << std::endl;
    std::cout << "----------" << std::endl;
    p2.fetch();
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << "p1[CURR].bits="; p1[CURR].print_bits();
    std::cout << "p1[CURR].acts="; p1[CURR].print_acts();
    std::cout << "p1[PREV].bits="; p1[PREV].print_bits();
    std::cout << "p1[PREV].acts="; p1[PREV].print_acts();
    std::cout << "p2[CURR].bits="; p2[CURR].print_bits();
    std::cout << "p2[CURR].acts="; p2[CURR].print_acts();
    std::cout << "p2[PREV].bits="; p2[PREV].print_bits();
    std::cout << "p2[PREV].acts="; p2[PREV].print_acts();
    std::cout << std::endl;

    std::cout << "Page Step" << std::endl;
    std::cout << "---------" << std::endl;
    p0.step();
    p1.step();
    p2.step();
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << "p1[CURR].bits="; p1[CURR].print_bits();
    std::cout << "p1[CURR].acts="; p1[CURR].print_acts();
    std::cout << "p1[PREV].bits="; p1[PREV].print_bits();
    std::cout << "p1[PREV].acts="; p1[PREV].print_acts();
    std::cout << "p2[CURR].bits="; p2[CURR].print_bits();
    std::cout << "p2[CURR].acts="; p2[CURR].print_acts();
    std::cout << "p2[PREV].bits="; p2[PREV].print_bits();
    std::cout << "p2[PREV].acts="; p2[PREV].print_acts();
    std::cout << std::endl;

    std::cout << "Page Compute Changed" << std::endl;
    std::cout << "--------------------" << std::endl;
    p0.compute_changed();
    p1.compute_changed();
    p2.compute_changed();
    std::cout << "p0.has_changed()=" << p0.has_changed() << std::endl;
    std::cout << "p1.has_changed()=" << p1.has_changed() << std::endl;
    std::cout << "p2.has_changed()=" << p2.has_changed() << std::endl;
    std::cout << std::endl;

    std::cout << "Page Copy Previous to Current" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    p0.copy_previous_to_current();
    p1.copy_previous_to_current();
    p2.copy_previous_to_current();
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << "p1[CURR].bits="; p1[CURR].print_bits();
    std::cout << "p1[CURR].acts="; p1[CURR].print_acts();
    std::cout << "p1[PREV].bits="; p1[PREV].print_bits();
    std::cout << "p1[PREV].acts="; p1[PREV].print_acts();
    std::cout << "p2[CURR].bits="; p2[CURR].print_bits();
    std::cout << "p2[CURR].acts="; p2[CURR].print_acts();
    std::cout << "p2[PREV].bits="; p2[PREV].print_bits();
    std::cout << "p2[PREV].acts="; p2[PREV].print_acts();
    std::cout << std::endl;

    std::cout << "Page Compute Changed (again)" << std::endl;
    std::cout << "----------------------------" << std::endl;
    p0.compute_changed();
    p1.compute_changed();
    p2.compute_changed();
    std::cout << "p0.has_changed()=" << p0.has_changed() << std::endl;
    std::cout << "p1.has_changed()=" << p1.has_changed() << std::endl;
    std::cout << "p2.has_changed()=" << p2.has_changed() << std::endl;
    std::cout << std::endl;

    std::cout << "Page Clear Bits" << std::endl;
    std::cout << "---------------" << std::endl;
    p0[CURR].clear_bits();
    p0[PREV].clear_bits();
    p1[CURR].clear_bits();
    p1[PREV].clear_bits();
    p2[CURR].clear_bits();
    p2[PREV].clear_bits();
    std::cout << "p0[CURR].bits="; p0[CURR].print_bits();
    std::cout << "p0[CURR].acts="; p0[CURR].print_acts();
    std::cout << "p0[PREV].bits="; p0[PREV].print_bits();
    std::cout << "p0[PREV].acts="; p0[PREV].print_acts();
    std::cout << "p1[CURR].bits="; p1[CURR].print_bits();
    std::cout << "p1[CURR].acts="; p1[CURR].print_acts();
    std::cout << "p1[PREV].bits="; p1[PREV].print_bits();
    std::cout << "p1[PREV].acts="; p1[PREV].print_acts();
    std::cout << "p2[CURR].bits="; p2[CURR].print_bits();
    std::cout << "p2[CURR].acts="; p2[CURR].print_acts();
    std::cout << "p2[PREV].bits="; p2[PREV].print_bits();
    std::cout << "p2[PREV].acts="; p2[PREV].print_acts();
    std::cout << std::endl;
}

#endif