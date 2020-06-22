#ifndef TEST_PATTERN_CLASSIFIER_H
#define TEST_PATTERN_CLASSIFIER_H

#include "pattern_classifier.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_pattern_classifier() {
    printf("================================================================================\n");
    printf("Test PatternClassifier\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    struct ScalarEncoder e;
    struct PatternClassifier pc;

    uint32_t* labels = malloc(2 * sizeof(*labels));
    labels[0] = 0;
    labels[1] = 1;

    printf("pattern_classifier_construct()\n");
    scalar_encoder_construct(&e, 0.0, 1.0, 128, 32);
    pattern_classifier_construct(&pc, labels, 2, 128, 2, 20, 2, 1, 0.8, 0.5, 0.25);
    page_add_child(pc.input, e.output);
    printf("\n");

    printf("printing parameters:\n");
    scalar_encoder_print_parameters(&e);
    pattern_classifier_print_parameters(&pc);
    printf("\n");


    printf("pattern_classifier_compute(label=0, learn=False)\n");
    scalar_encoder_compute(&e, 0);
    pattern_classifier_compute(&pc, 0, 0);
    printf(" e bits=");
    page_print_bits(pc.input, CURR);
    printf("pc bits=");
    page_print_bits(pc.output, CURR);
    printf("pc labs=");
    pattern_classifier_print_state_labels(&pc);
    printf("pc prob=");
    pattern_classifier_print_probabilities(&pc);
    printf("\n");

    printf("pattern_classifier_compute(label=1, learn=False)\n");
    scalar_encoder_compute(&e, 1);
    pattern_classifier_compute(&pc, 1, 0);
    printf(" e bits=");
    page_print_bits(pc.input, CURR);
    printf("pc bits=");
    page_print_bits(pc.output, CURR);
    printf("pc labs=");
    pattern_classifier_print_state_labels(&pc);
    printf("pc prob=");
    pattern_classifier_print_probabilities(&pc);
    printf("\n");

    printf("learning 10 steps...\n");
    for (uint32_t i = 0; i < 5; i++) {
        scalar_encoder_compute(&e, 0);
        pattern_classifier_compute(&pc, 0, 1);
        scalar_encoder_compute(&e, 1);
        pattern_classifier_compute(&pc, 1, 1);
    }
    printf("\n");

    printf("pattern_classifier_compute(label=0, learn=False)\n");
    scalar_encoder_compute(&e, 0);
    pattern_classifier_compute(&pc, 0, 0);
    printf(" e bits=");
    page_print_bits(pc.input, CURR);
    printf("pc bits=");
    page_print_bits(pc.output, CURR);
    printf("pc labs=");
    pattern_classifier_print_state_labels(&pc);
    printf("pc prob=");
    pattern_classifier_print_probabilities(&pc);
    printf("\n");

    printf("pattern_classifier_compute(label=1, learn=False)\n");
    scalar_encoder_compute(&e, 1);
    pattern_classifier_compute(&pc, 1, 0);
    printf(" e bits=");
    page_print_bits(pc.input, CURR);
    printf("pc bits=");
    page_print_bits(pc.output, CURR);
    printf("pc labs=");
    pattern_classifier_print_state_labels(&pc);
    printf("pc prob=");
    pattern_classifier_print_probabilities(&pc);
    printf("\n");

    printf("pattern_classifier_destruct()\n");
    scalar_encoder_destruct(&e);
    pattern_classifier_destruct(&pc);
    free(labels);
}

#endif