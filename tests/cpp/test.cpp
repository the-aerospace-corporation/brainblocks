#include "utils.hpp"
#include "test_bitarray.hpp"
#include "test_page.hpp"
#include "test_coincidence_set.hpp"
#include "test_blank_block.hpp"
#include "test_scalar_encoder.hpp"
#include "test_symbols_encoder.hpp"
#include "test_persistence_encoder.hpp"
#include "test_pattern_classifier.hpp"
#include "test_pattern_pooler.hpp"
#include "test_sequence_learner.hpp"

int main() {
    test_bitarray();
    test_page();
    test_coincidence_set();
    test_blank_block();
    test_scalar_encoder();
    test_symbols_encoder();
    test_persistence_encoder();
    test_pattern_classifier();
    test_pattern_pooler();
    test_sequence_learner();
}