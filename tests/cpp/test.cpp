#include "test_bitarray.hpp"
#include "test_page.hpp"
#include "test_blank_block.hpp"

int main() {
    test_bitarray();
    test_page();
    //test_coincidence_set();
    test_blank_block();

    //test_scalar_encoder();
    //test_persistence_encoder();
    //test_pattern_classifier();
    //test_pattern_pooler();
    //test_sequence_learner();
    //test_sequence_learner_pooled();
    //test_sequence_learner_pooled_timed();
}