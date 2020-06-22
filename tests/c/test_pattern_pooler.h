#ifndef TEST_PATTERN_POOLER_H
#define TEST_PATTERN_POOLER_H

#include "pattern_pooler.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_pattern_pooler() {
    printf("================================================================================\n");
    printf("Test PatternPooler\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    struct ScalarEncoder e;
    struct PatternPooler pp;

    printf("pattern_pooler_construct()\n");
    scalar_encoder_construct(&e, 0.0, 1.0, 128, 16);
    pattern_pooler_construct(&pp, 128, 2, 20, 2, 1, 0.8, 0.8, 0.25);
    page_add_child(pp.input, e.output);
    printf("\n");

    printf("printing parameters:\n");
    scalar_encoder_print_parameters(&e);
    pattern_pooler_print_parameters(&pp);
    printf("\n");

    printf("pattern_pooler_compute(input=0.0, learn=0)\n");
    scalar_encoder_compute(&e, 0.0);
    pattern_pooler_compute(&pp, 0);
    printf(" e bits=");
    page_print_bits(pp.input, CURR);
    printf(" e acts=");
    page_print_acts(pp.input, CURR);
    printf("pp bits=");
    page_print_bits(pp.output, CURR);
    printf("pp acts=");
    page_print_acts(pp.output, CURR);
    printf("pp over=");
    pattern_pooler_print_overlaps(&pp);
    printf("\n");

    printf("pattern_pooler_compute(input=1.0, learn=0)\n");
    scalar_encoder_compute(&e, 1.0);
    pattern_pooler_compute(&pp, 0);
    printf(" e bits=");
    page_print_bits(pp.input, CURR);
    printf(" e acts=");
    page_print_acts(pp.input, CURR);
    printf("pp bits=");
    page_print_bits(pp.output, CURR);
    printf("pp acts=");
    page_print_acts(pp.output, CURR);
    printf("pp over=");
    pattern_pooler_print_overlaps(&pp);
    printf("\n");

    printf("learning 10 steps...\n");
    for (uint32_t i = 0; i < 5; i++) {
        scalar_encoder_compute(&e, 0.0);
        pattern_pooler_compute(&pp, 1);
        scalar_encoder_compute(&e, 1.0);
        pattern_pooler_compute(&pp, 1);
    }
    printf("\n");

    printf("pattern_pooler_compute(input=0.0, learn=0)\n");
    scalar_encoder_compute(&e, 0.0);
    pattern_pooler_compute(&pp, 0);
    printf(" e bits=");
    page_print_bits(pp.input, CURR);
    printf(" e acts=");
    page_print_acts(pp.input, CURR);
    printf("pp bits=");
    page_print_bits(pp.output, CURR);
    printf("pp acts=");
    page_print_acts(pp.output, CURR);
    printf("pp over=");
    pattern_pooler_print_overlaps(&pp);
    printf("\n");

    printf("pattern_pooler_compute(input=1.0, learn=0)\n");
    scalar_encoder_compute(&e, 1.0);
    pattern_pooler_compute(&pp, 0);
    printf(" e bits=");
    page_print_bits(pp.input, CURR);
    printf(" e acts=");
    page_print_acts(pp.input, CURR);
    printf("pp bits=");
    page_print_bits(pp.output, CURR);
    printf("pp acts=");
    page_print_acts(pp.output, CURR);
    printf("pp over=");
    pattern_pooler_print_overlaps(&pp);
    printf("\n");

    printf("pattern_pooler_construct()\n");
    scalar_encoder_destruct(&e);
    pattern_pooler_destruct(&pp);
}

#endif