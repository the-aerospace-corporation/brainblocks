// =============================================================================
// test_sequence_learner.cpp
// =============================================================================
#include "blocks/sequence_learner.hpp"
#include "blocks/scalar_transformer.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace BrainBlocks;

int main() {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::duration<double> duration;

    std::vector<double> values = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    //std::vector<double> values = {
    //    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    //    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    //    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<double> scores(values.size());

    // Setup blocks
    ScalarTransformer st(0.0, 1.0, 512, 8, 2);
    SequenceLearner sl(512, 10, 10, 12, 6, 20, 2, 1, 2);

    // Setup block connetions
    sl.input.add_child(&st.output, CURR);

    // Initialize blocks
    sl.init();

    // Compute loop
    for (uint32_t i = 0; i < values.size(); i++) {

        // Compute scalar transformer
        st.set_value(values[i]);
        st.feedforward();

        // Compute sequence learner
        t0 = std::chrono::high_resolution_clock::now();
        sl.feedforward(true);
        t1 = std::chrono::high_resolution_clock::now();
        duration = t1 - t0;
        std::cout << "t=" << duration.count() << "s" << std::endl;

    scores[i] = sl.get_anomaly_score();
    }

    // Print results
    std::cout << "values, scores" << std::endl;
    std::cout << std::fixed;
    for (uint32_t i = 0; i < scores.size(); i++) {
        std::cout << std::setprecision(4) << values[i] << ", " << scores[i]
        << std::endl;
    }

    return 0;
}
