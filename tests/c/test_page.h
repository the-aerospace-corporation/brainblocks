#ifndef TEST_PAGE_H
#define TEST_PAGE_H

#include "page.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_page() {
    printf("================================================================================\n");
    printf("Test Page\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    struct Page p0;
    struct Page p1;
    struct Page p2;

    printf("page_construct()\n");
    page_construct(&p0, 2, 32);
    page_construct(&p1, 2, 32);
    page_construct(&p2, 2, 0);

    printf("page_add_child()\n");
    page_add_child(&p2, &p0);
    page_add_child(&p2, &p1);
    printf("\n");

    printf("page_initialize()\n");
    page_initialize(&p0);
    page_initialize(&p1);
    page_initialize(&p2);
    printf("p0 bits[CURR]=");
    page_print_bits(&p0, CURR);
    printf("p0 acts[CURR]=");
    page_print_acts(&p0, CURR);
    printf("p0 bits[PREV]=");
    page_print_bits(&p0, PREV);
    printf("p0 acts[PREV]=");
    page_print_acts(&p0, PREV);
    printf("p1 bits[CURR]=");
    page_print_bits(&p1, CURR);
    printf("p1 acts[CURR]=");
    page_print_acts(&p1, CURR);
    printf("p1 bits[PREV]=");
    page_print_bits(&p1, PREV);
    printf("p1 acts[PREV]=");
    page_print_acts(&p1, PREV);
    printf("p2 bits[CURR]=");
    page_print_bits(&p2, CURR);
    printf("p2 acts[CURR]=");
    page_print_acts(&p2, CURR);
    printf("p2 bits[PREV]=");
    page_print_bits(&p2, PREV);
    printf("p2 acts[PREV]=");
    page_print_acts(&p2, PREV);
    printf("\n");

    printf("page_set_bit()\n");
    page_set_bit(&p0, 0, 0);
    page_set_bit(&p0, 0, 2);
    page_set_bit(&p0, 0, 3);
    page_set_bit(&p0, 0, 4);
    page_set_bit(&p0, 0, 5);
    page_set_bit(&p0, 0, 6);
    page_set_bit(&p0, 0, 31);
    page_set_bit(&p1, 0, 0);
    page_set_bit(&p1, 0, 16);
    page_set_bit(&p1, 0, 18);
    page_set_bit(&p1, 0, 20);
    page_set_bit(&p1, 0, 22);
    page_set_bit(&p1, 0, 31);
    printf("p0 bits[CURR]=");
    page_print_bits(&p0, CURR);
    printf("p0 acts[CURR]=");
    page_print_acts(&p0, CURR);
    printf("p0 bits[PREV]=");
    page_print_bits(&p0, PREV);
    printf("p0 acts[PREV]=");
    page_print_acts(&p0, PREV);
    printf("p1 bits[CURR]=");
    page_print_bits(&p1, CURR);
    printf("p1 acts[CURR]=");
    page_print_acts(&p1, CURR);
    printf("p1 bits[PREV]=");
    page_print_bits(&p1, PREV);
    printf("p1 acts[PREV]=");
    page_print_acts(&p1, PREV);
    printf("p2 bits[CURR]=");
    page_print_bits(&p2, CURR);
    printf("p2 acts[CURR]=");
    page_print_acts(&p2, CURR);
    printf("p2 bits[PREV]=");
    page_print_bits(&p2, PREV);
    printf("p2 acts[PREV]=");
    page_print_acts(&p2, PREV);
    printf("\n");

    printf("page_fetch()\n");
    page_fetch(&p2);
    printf("p0 bits[CURR]=");
    page_print_bits(&p0, CURR);
    printf("p0 acts[CURR]=");
    page_print_acts(&p0, CURR);
    printf("p0 bits[PREV]=");
    page_print_bits(&p0, PREV);
    printf("p0 acts[PREV]=");
    page_print_acts(&p0, PREV);
    printf("p1 bits[CURR]=");
    page_print_bits(&p1, CURR);
    printf("p1 acts[CURR]=");
    page_print_acts(&p1, CURR);
    printf("p1 bits[PREV]=");
    page_print_bits(&p1, PREV);
    printf("p1 acts[PREV]=");
    page_print_acts(&p1, PREV);
    printf("p2 bits[CURR]=");
    page_print_bits(&p2, CURR);
    printf("p2 acts[CURR]=");
    page_print_acts(&p2, CURR);
    printf("p2 bits[PREV]=");
    page_print_bits(&p2, PREV);
    printf("p2 acts[PREV]=");
    page_print_acts(&p2, PREV);
    printf("\n");

    printf("page_step()\n");
    page_step(&p0);
    page_step(&p1);
    page_step(&p2);
    printf("p0 bits[CURR]=");
    page_print_bits(&p0, CURR);
    printf("p0 acts[CURR]=");
    page_print_acts(&p0, CURR);
    printf("p0 bits[PREV]=");
    page_print_bits(&p0, PREV);
    printf("p0 acts[PREV]=");
    page_print_acts(&p0, PREV);
    printf("p1 bits[CURR]=");
    page_print_bits(&p1, CURR);
    printf("p1 acts[CURR]=");
    page_print_acts(&p1, CURR);
    printf("p1 bits[PREV]=");
    page_print_bits(&p1, PREV);
    printf("p1 acts[PREV]=");
    page_print_acts(&p1, PREV);
    printf("p2 bits[CURR]=");
    page_print_bits(&p2, CURR);
    printf("p2 acts[CURR]=");
    page_print_acts(&p2, CURR);
    printf("p2 bits[PREV]=");
    page_print_bits(&p2, PREV);
    printf("p2 acts[PREV]=");
    page_print_acts(&p2, PREV);
    printf("\n");

    printf("page_compute_changed()\n");
    page_compute_changed(&p0);
    page_compute_changed(&p1);
    page_compute_changed(&p2);
    printf("p0 changed_flag=%i\n", p0.changed_flag);
    printf("p1 changed_flag=%i\n", p1.changed_flag);
    printf("p2 changed_flag=%i\n", p2.changed_flag);
    printf("\n");

    printf("page_copy_previous_to_current()\n");
    page_copy_previous_to_current(&p0);
    page_copy_previous_to_current(&p1);
    page_copy_previous_to_current(&p2);
    printf("p0 bits[CURR]=");
    page_print_bits(&p0, CURR);
    printf("p0 acts[CURR]=");
    page_print_acts(&p0, CURR);
    printf("p0 bits[PREV]=");
    page_print_bits(&p0, PREV);
    printf("p0 acts[PREV]=");
    page_print_acts(&p0, PREV);
    printf("p1 bits[CURR]=");
    page_print_bits(&p1, CURR);
    printf("p1 acts[CURR]=");
    page_print_acts(&p1, CURR);
    printf("p1 bits[PREV]=");
    page_print_bits(&p1, PREV);
    printf("p1 acts[PREV]=");
    page_print_acts(&p1, PREV);
    printf("p2 bits[CURR]=");
    page_print_bits(&p2, CURR);
    printf("p2 acts[CURR]=");
    page_print_acts(&p2, CURR);
    printf("p2 bits[PREV]=");
    page_print_bits(&p2, PREV);
    printf("p2 acts[PREV]=");
    page_print_acts(&p2, PREV);
    printf("\n");

    printf("page_compute_changed()\n");
    page_compute_changed(&p0);
    page_compute_changed(&p1);
    page_compute_changed(&p2);
    printf("p0 changed_flag=%i\n", p0.changed_flag);
    printf("p1 changed_flag=%i\n", p1.changed_flag);
    printf("p2 changed_flag=%i\n", p2.changed_flag);
    printf("\n");

    printf("page_clear_bits()\n");
    page_clear_bits(&p0, 0);
    page_clear_bits(&p1, 0);
    page_clear_bits(&p2, 0);
    printf("p0 bits[CURR]=");
    page_print_bits(&p0, CURR);
    printf("p0 acts[CURR]=");
    page_print_acts(&p0, CURR);
    printf("p0 bits[PREV]=");
    page_print_bits(&p0, PREV);
    printf("p0 acts[PREV]=");
    page_print_acts(&p0, PREV);
    printf("p1 bits[CURR]=");
    page_print_bits(&p1, CURR);
    printf("p1 acts[CURR]=");
    page_print_acts(&p1, CURR);
    printf("p1 bits[PREV]=");
    page_print_bits(&p1, PREV);
    printf("p1 acts[PREV]=");
    page_print_acts(&p1, PREV);
    printf("p2 bits[CURR]=");
    page_print_bits(&p2, CURR);
    printf("p2 acts[CURR]=");
    page_print_acts(&p2, CURR);
    printf("p2 bits[PREV]=");
    page_print_bits(&p2, PREV);
    printf("p2 acts[PREV]=");
    page_print_acts(&p2, PREV);
    printf("\n");

    printf("page_destruct()\n");
    page_destruct(&p0);
    page_destruct(&p1);
    page_destruct(&p2);
}

#endif