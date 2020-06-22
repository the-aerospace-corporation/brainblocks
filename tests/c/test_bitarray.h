#ifndef TEST_BITARRAY_H
#define TEST_BITARRAY_H

#include "bitarray.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_bitarray() {
    printf("================================================================================\n");
    printf("Test BitArray\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    struct BitArray ba0;
    struct BitArray ba1;
    struct BitArray ba2;

    printf("bitarray_construct()\n");
    bitarray_construct(&ba0, 32);
    bitarray_construct(&ba1, 64);
    bitarray_construct(&ba2, 64);
    bitarray_print_bits(&ba0);
    printf("\n");

    printf("bitarray_resize()\n");
    bitarray_resize(&ba0, 64);
    bitarray_print_bits(&ba0);
    printf("\n");

    printf("bitarray_set_bit()\n");
    bitarray_set_bit(&ba0, 4);
    bitarray_print_bits(&ba0);
    printf("\n");

    printf("bitarray_get_bit()\n");
    uint32_t bit3 = bitarray_get_bit(&ba0, 3);
    uint32_t bit4 = bitarray_get_bit(&ba0, 4);
    uint32_t bit5 = bitarray_get_bit(&ba0, 5);
    bitarray_print_bits(&ba0);
    printf("bit[3]=%i\n", bit3);
    printf("bit[4]=%i\n", bit4);
    printf("bit[5]=%i\n", bit5);
    printf("\n");

    printf("bitarray_clear_bit()\n");
    bitarray_clear_bit(&ba0, 4);
    bitarray_print_bits(&ba0);
    printf("\n");

    printf("bitarray_set()\n");
    bitarray_set(&ba0, 2, 7);
    bitarray_print_bits(&ba0);
    printf("\n");

    printf("bitarray_clear()\n");
    bitarray_clear(&ba0);
    bitarray_print_bits(&ba0);
    printf("\n");

    printf("bitarray_random_fill()\n");
    bitarray_random_fill(&ba0, 0.5);
    bitarray_print_bits(&ba0);
    printf("\n");

    printf("bitarray_copy()\n");
    printf("before src=");
    bitarray_print_bits(&ba0);
    printf("before dst=");
    bitarray_print_bits(&ba1);
    bitarray_copy(&ba1, &ba0, 0, 0, 2);
    printf(" after src=");
    bitarray_print_bits(&ba0);
    printf(" after dst=");
    bitarray_print_bits(&ba1);
    printf("\n");

    bitarray_clear(&ba1);

    printf("bitarray_copy() // offset and subset\n");
    printf("before src=");
    bitarray_print_bits(&ba0);
    printf("before dst=");
    bitarray_print_bits(&ba1);
    bitarray_copy(&ba1, &ba0, 1, 0, 1);
    printf(" after src=");
    bitarray_print_bits(&ba0);
    printf(" after dst=");
    bitarray_print_bits(&ba1);
    printf("\n");

    bitarray_clear(&ba0);
    bitarray_clear(&ba1);
    bitarray_random_fill(&ba0, 0.5);
    bitarray_random_fill(&ba1, 0.5);

    printf("bitarray_not()\n");
    bitarray_not(&ba0, &ba2);
    printf(" in=");
    bitarray_print_bits(&ba0);
    printf("out=");
    bitarray_print_bits(&ba2);
    printf("\n");

    printf("bitarray_and()\n");
    bitarray_clear(&ba2);
    bitarray_and(&ba0, &ba1, &ba2);
    printf("in0=");
    bitarray_print_bits(&ba0);
    printf("in1=");
    bitarray_print_bits(&ba1);
    printf("out=");
    bitarray_print_bits(&ba2);
    printf("\n");

    printf("bitarray_or()\n");
    bitarray_clear(&ba2);
    bitarray_or(&ba0, &ba1, &ba2);
    printf("in0=");
    bitarray_print_bits(&ba0);
    printf("in1=");
    bitarray_print_bits(&ba1);
    printf("out=");
    bitarray_print_bits(&ba2);
    printf("\n");

    printf("bitarray_xor()\n");
    bitarray_clear(&ba2);
    bitarray_xor(&ba0, &ba1, &ba2);
    printf("in0=");
    bitarray_print_bits(&ba0);
    printf("in1=");
    bitarray_print_bits(&ba1);
    printf("out=");
    bitarray_print_bits(&ba2);
    printf("\n");

    printf("bitarray_count()\n");
    uint32_t count = bitarray_count(&ba0);
    bitarray_print_bits(&ba0);
    printf("count=%i\n", count);
    printf("\n");

    printf("bitarray_get_actarray()\n");
    bitarray_get_actarray(&ba0);
    bitarray_print_acts(&ba0);
    printf("\n");

    printf("bitarray_destruct()\n");
    bitarray_destruct(&ba0);
    bitarray_destruct(&ba1);
    bitarray_destruct(&ba2);
}

#endif