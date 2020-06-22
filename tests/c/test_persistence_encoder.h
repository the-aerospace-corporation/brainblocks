#ifndef TEST_PERSISTENCE_ENCODER_H
#define TEST_PERSISTENCE_ENCODER_H

#include "persistence_encoder.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_persistence_encoder() {
    printf("================================================================================\n");
    printf("Test PersistenceEncoder\n");
    printf("================================================================================\n");
    printf("\n");
    
    utils_seed(0);

    struct PersistenceEncoder e;
    
    printf("persistence_encoder_construct()\n");
    persistence_encoder_construct(&e, -1.0, 1.0, 64, 8, 4);
    printf("\n");

    printf("printing parameters:\n");
    persistence_encoder_print_parameters(&e);
    printf("\n");

    printf("persistence_encoder_reset()\n");
    persistence_encoder_reset(&e);
    printf("\n");

    printf("persistence_encoder_compute(0.0)\n");
    persistence_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("persistence_encoder_compute(0.0)\n");
    persistence_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("persistence_encoder_compute(0.0)\n");
    persistence_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("persistence_encoder_compute(0.0)\n");
    persistence_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("persistence_encoder_compute(0.0)\n");
    persistence_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("persistence_encoder_compute(0.0)\n");
    persistence_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("persistence_encoder_compute(0.0)\n");
    persistence_encoder_compute(&e, 0.0);
    printf("bits=");
    page_print_bits(e.output, CURR);
    printf("acts=");
    page_print_acts(e.output, CURR);
    printf("\n");

    printf("persistence_encoder_destruct()\n");
    persistence_encoder_destruct(&e);
}

#endif