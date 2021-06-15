#include "blocks/pattern_classifier_dynamic.hpp"
#include "blocks/scalar_transformer.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace BrainBlocks;

int main() {

    //std::chrono::high_resolution_clock::time_point t0;
    //std::chrono::high_resolution_clock::time_point t1;
    //std::chrono::duration<double> duration;

    ScalarTransformer st(0.0, 1.0, 1024, 128);
    PatternClassifierDynamic pc(1024, 128, 8, 20, 2, 1, 0.8, 0.5, 0.3, 2);

    pc.input.add_child(&st.output, 0);

    pc.init();

    std::vector<double> probs;

    for (uint32_t i = 0; i < 10; i++) {
        st.set_value(0.0);
        pc.set_label(0);
        st.feedforward();
        pc.feedforward(true);
        probs = pc.get_probabilities();
        std::cout << "{" << probs[0] << ", " << probs[1] << "}" << std::endl;
    }

    std::cout << std::endl;

    for (uint32_t i = 0; i < 10; i++) {
        st.set_value(1.0);
        pc.set_label(1);
        st.feedforward();
        pc.feedforward(true);
        probs = pc.get_probabilities();
        std::cout << "{" << probs[0] << ", " << probs[1] << "}" << std::endl;
    }

    std::cout << std::endl;

    st.set_value(0.0);
    st.feedforward();
    pc.feedforward(false);
    probs = pc.get_probabilities();
    std::cout << "{" << probs[0] << ", " << probs[1] << "}" << std::endl;
    std::cout << std::endl;

    st.set_value(1.0);
    st.feedforward();
    pc.feedforward(false);
    probs = pc.get_probabilities();
    std::cout << "{" << probs[0] << ", " << probs[1] << "}" << std::endl;
    std::cout << std::endl;

    return 0;
}
