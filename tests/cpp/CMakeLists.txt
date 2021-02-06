# ==============================================================================
# tests/cpp/CMakeLists.txt
# ==============================================================================

include_directories(${BRAINBLOCKS_SOURCE_DIR}/src/cpp)

add_executable(test_bitarray test_bitarray.cpp)
add_executable(test_block_input test_block_input.cpp)
add_executable(test_block_memory test_block_memory.cpp)
add_executable(test_block_output test_block_output.cpp)
add_executable(test_context_learner test_context_learner.cpp)
add_executable(test_discrete_transformer test_discrete_transformer.cpp)
add_executable(test_pattern_classifier test_pattern_classifier.cpp)
add_executable(test_pattern_classifier_dynamic
               test_pattern_classifier_dynamic.cpp)
add_executable(test_pattern_pooler test_pattern_pooler.cpp)
add_executable(test_persistence_transformer test_persistence_transformer.cpp)
add_executable(test_scalar_transformer test_scalar_transformer.cpp)
add_executable(test_sequence_learner test_sequence_learner.cpp)

target_link_libraries(test_bitarray bbcore)
target_link_libraries(test_block_input bbcore)
target_link_libraries(test_block_memory bbcore)
target_link_libraries(test_block_output bbcore)
target_link_libraries(test_context_learner bbcore)
target_link_libraries(test_discrete_transformer bbcore)
target_link_libraries(test_pattern_classifier bbcore)
target_link_libraries(test_pattern_classifier_dynamic bbcore)
target_link_libraries(test_pattern_pooler bbcore)
target_link_libraries(test_persistence_transformer bbcore)
target_link_libraries(test_scalar_transformer bbcore)
target_link_libraries(test_sequence_learner bbcore)
