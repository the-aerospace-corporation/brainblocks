#include "wrappers/class_interfaces.hpp"

#include <chrono>
#include <ctime>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void test_sequence_learner() {

    double data[35] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0};

    double scores[35];
    double rtimes[35];

    for (uint32_t i = 0; i < 35; i++) {
        scores[i] = 0.0;
        rtimes[i] = 0.0;
    }

    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;

    ScalarEncoderClass *e = new ScalarEncoderClass(0.0, 1.0, 1024, 128);
    PatternPoolerClass *pp = new PatternPoolerClass(512, 8, 20, 2, 1, 0.8, 0.5, 0.25);
    SequenceLearnerClass *sl = new SequenceLearnerClass(10, 10, 12, 6, 1, 1, 0);

    pp->get_input()->add_child(e->get_output());
    sl->get_input()->add_child(pp->get_output());

    for (uint32_t i = 0; i < 35; i++) {
        m_StartTime = std::chrono::system_clock::now();
        e->compute(data[i]);
        pp->compute(1);
        sl->compute(1);
        scores[i] = sl->get_score();
        m_EndTime = std::chrono::system_clock::now();
        rtimes[i] = std::chrono::duration_cast<std::chrono::microseconds>(m_EndTime - m_StartTime).count() / 1000000.0;
    }

    printf("values,   scores,   rtimes\n");
    for (uint32_t i = 0; i < 35; i++) {
        printf("%f, %f, %fs\n", data[i], scores[i], rtimes[i]);
    }

    delete e;
    delete pp;
    delete sl;
}

// ================================================================================
// main
// ================================================================================
int main() {
    srand((uint32_t)time(NULL));

    test_sequence_learner();

    return 0;
}