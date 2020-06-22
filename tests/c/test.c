#include "test_bitarray.h"
#include "test_page.h"
#include "test_coincidence_set.h"
#include "test_blank_block.h"
#include "test_scalar_encoder.h"
#include "test_persistence_encoder.h"
#include "test_pattern_classifier.h"
#include "test_pattern_pooler.h"
#include "test_sequence_learner.h"

void main() {
    test_bitarray();
    test_page();
    test_coincidence_set();
    test_blank_block();
    test_scalar_encoder();
    test_persistence_encoder();
    test_pattern_classifier();
    test_pattern_pooler();
    test_sequence_learner();
    test_sequence_learner_pooled();
    test_sequence_learner_pooled_timed();
}