#ifndef TEST_SCALAR_ENCODER_H
#define TEST_SCALAR_ENCODER_H

#include "scalar_encoder.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_scalar_encoder() {
    printf("================================================================================\n");
    printf("Test ScalarEncoder\n");
    printf("================================================================================\n");
    printf("\n");
    
    utils_seed(0);

    struct ScalarEncoder e;
    
    printf("scalar_encoder_construct()\n");
    scalar_encoder_construct(&e, -1.0, 1.0, 64, 8);
    printf("\n");

    printf("printing parameters:\n");
    scalar_encoder_print_parameters(&e);
    printf("\n");

    printf("scalar_encoder_compute(-1.5)\n");
    scalar_encoder_compute(&e, -1.5);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("scalar_encoder_compute(-1.0)\n");
    scalar_encoder_compute(&e, -1.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("scalar_encoder_compute(-0.5)\n");
    scalar_encoder_compute(&e, -0.5);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("scalar_encoder_compute(0.0)\n");
    scalar_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("scalar_encoder_compute(0.5)\n");
    scalar_encoder_compute(&e, 0.5);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("scalar_encoder_compute(1.0)\n");
    scalar_encoder_compute(&e, 1.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("scalar_encoder_compute(1.5)\n");
    scalar_encoder_compute(&e, 1.5);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("scalar_encoder_destruct()\n");
    scalar_encoder_destruct(&e);
}

#endif