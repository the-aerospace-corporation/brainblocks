#ifndef TEST_INDICATOR_H
#define TEST_INDICATOR_H

#include "coincidence_set.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_coincidence_set() {
    printf("================================================================================\n");
    printf("Test CoincidenceSet\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    uint32_t NUM_I = 64;
    uint32_t NUM_AI = 16;
    uint32_t NUM_R = 10;
    uint32_t PERM_THR = 5;

    uint32_t* learn_mask = calloc(NUM_AI, sizeof(*learn_mask));
    struct BitArray input_ba;
    struct CoincidenceSet r;

    bitarray_construct(&input_ba, NUM_I);

    for (uint32_t i = 0; i < NUM_AI; i++) {
        bitarray_set_bit(&input_ba, i);
    }

    for (uint32_t i = 0; i < NUM_R; i++) {
        learn_mask[i] = 1;
    }

    printf("coincidence_set_construct_pooled()\n");
    coincidence_set_construct_pooled(&r, NUM_I, NUM_R, (uint32_t)(NUM_R/2), PERM_THR);
    printf("r=");
    coincidence_set_print(&r);
    printf("\n");

    printf("coincidence_set_overlap()\n");
    coincidence_set_overlap(&r, &input_ba);
    printf("in=");
    bitarray_print_acts(&input_ba);
    printf("r=");
    coincidence_set_print(&r);
    printf("overlap=%i\n", r.overlap);
    printf("\n");

    printf("coincidence_set_learn()\n");
    coincidence_set_learn(&r, &input_ba, learn_mask, 1, 1);
    coincidence_set_update_connections(&r, PERM_THR);
    printf("in=");
    bitarray_print_acts(&input_ba);
    printf("r=");
    coincidence_set_print(&r);
    printf("\n");

    printf("coincidence_set_learn()\n");
    coincidence_set_learn(&r, &input_ba, learn_mask, 100, 100);
    coincidence_set_update_connections(&r, PERM_THR);
    printf("in=");
    bitarray_print_acts(&input_ba);
    printf("r=");
    coincidence_set_print(&r);
    printf("\n");

    printf("coincidence_set_destruct()\n");
    free(learn_mask);
    bitarray_destruct(&input_ba);
    coincidence_set_destruct(&r);
}

#endif