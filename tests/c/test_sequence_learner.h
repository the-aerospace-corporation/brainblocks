#ifndef TEST_SEQUENCE_LEARNER_H
#define TEST_SEQUENCE_LEARNER_H

#include "sequence_learner.h"

#include "helper.h"
#include "utils.h"
#include "timer.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_sequence_learner() {
    printf("================================================================================\n");
    printf("Test SequenceLearner\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    struct ScalarEncoder e;
    struct SequenceLearner sl;

    double values[30] = {
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9};

    double scores[30];
    double rtimes[30];

    for (uint32_t i = 0; i < 30; i++) {
        scores[i] = 0.0;
        rtimes[i] = 0.0;
    }

    printf("constructing...\n");
    scalar_encoder_construct(&e, 0.0, 1.0, 64, 8);
    sequence_learner_construct(&sl, 10, 10, 12, 6, 1, 1, 0);
    page_add_child(sl.input, e.output);
    printf("\n");

    printf("printing parameters:\n");
    scalar_encoder_print_parameters(&e);
    sequence_learner_print_parameters(&sl);
    printf("\n");

    printf("learning...\n");
    for (uint32_t i = 0; i < 30; i++) {
        scalar_encoder_compute(&e, values[i]);
        sequence_learner_compute(&sl, 1);
        scores[i] = sequence_learner_get_score(&sl);
    }
    printf("\n");

    printf("values,   scores\n");
    for (uint32_t i = 0; i < 30; i++) {
        printf("%f, %f\n", values[i], scores[i]);
    }
    printf("\n");

    scalar_encoder_destruct(&e);
    sequence_learner_destruct(&sl);
}


void test_sequence_learner_pooled() {
    printf("================================================================================\n");
    printf("Test SequenceLearnerPooled\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    struct ScalarEncoder e;
    struct PatternPooler pp;
    struct SequenceLearner sl;

    double values[30] = {
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9};

    double scores[35];

    for (uint32_t i = 0; i < 30; i++) {
        scores[i] = 0.0;
    }

    printf("constructing...\n");
    scalar_encoder_construct(&e, 0.0, 1.0, 1024, 128);
    pattern_pooler_construct(&pp, 512, 8, 20, 2, 1, 0.8, 0.5, 0.25);
    sequence_learner_construct(&sl, 10, 10, 12, 6, 1, 1, 0);
    page_add_child(pp.input, e.output);
    page_add_child(sl.input, pp.output);
    printf("\n");

    printf("printing parameters:\n");
    scalar_encoder_print_parameters(&e);
    pattern_pooler_print_parameters(&pp);
    sequence_learner_print_parameters(&sl);
    printf("\n");

    printf("learning...\n");
    for (uint32_t i = 0; i < 30; i++) {
        scalar_encoder_compute(&e, values[i]);
        pattern_pooler_compute(&pp,1);
        sequence_learner_compute(&sl,1);
        scores[i] = sequence_learner_get_score(&sl);
    }
    printf("\n");

    printf("values,   scores\n");
    for (uint32_t i = 0; i < 30; i++) {
        printf("%f, %f\n", values[i], scores[i]);
    }
    printf("\n");

    scalar_encoder_destruct(&e);
    pattern_pooler_destruct(&pp);
    sequence_learner_destruct(&sl);
}


void test_sequence_learner_pooled_timed() {
    printf("================================================================================\n");
    printf("Test SequenceLearnerPooled Timed\n");
    printf("================================================================================\n");
    printf("\n");

    utils_seed(0);

    struct Timer t;
    timer_init(&t);

    struct ScalarEncoder e;
    struct PatternPooler pp;
    struct SequenceLearner sl;

    double values[30] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0};

    double scores[35];
    double rtimes[35];

    for (uint32_t i = 0; i < 30; i++) {
        scores[i] = 0.0;
        rtimes[i] = 0.0;
    }

    printf("constructing...\n");
    timer_start(&t);
    scalar_encoder_construct(&e, 0.0, 1.0, 1024, 128);
    pattern_pooler_construct(&pp, 512, 8, 20, 2, 1, 0.8, 0.5, 0.25);
    sequence_learner_construct(&sl, 10, 10, 12, 6, 1, 1, 0);
    page_add_child(pp.input, e.output);
    page_add_child(sl.input, pp.output);
    timer_stop(&t);
    printf("t=%fs\n", timer_get(&t));
    printf("\n");

    printf("printing parameters:\n");
    scalar_encoder_print_parameters(&e);
    pattern_pooler_print_parameters(&pp);
    sequence_learner_print_parameters(&sl);
    printf("\n");

    printf("learning...\n");
    for (uint32_t i = 0; i < 30; i++) {
        timer_start(&t);
        scalar_encoder_compute(&e, values[i]);
        pattern_pooler_compute(&pp,1);
        sequence_learner_compute(&sl,1);
        scores[i] = sequence_learner_get_score(&sl);
        timer_stop(&t);
        rtimes[i] = timer_get(&t);
    }
    printf("\n");

    printf("values,   scores,   rtimes\n");
    for (uint32_t i = 0; i < 30; i++) {
        printf("%f, %f, %fs\n", values[i], scores[i], rtimes[i]);
    }
    printf("\n");

    scalar_encoder_destruct(&e);
    pattern_pooler_destruct(&pp);
    sequence_learner_destruct(&sl);
}

#endif