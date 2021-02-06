# ==============================================================================
# src/cpp/CMakeLists.txt
# ==============================================================================

set(SOURCE_FILES
    bitarray.cpp
    block.cpp
    block_input.cpp
    block_memory.cpp
    block_output.cpp
    blocks/blank_block.cpp
    blocks/context_learner.cpp
    blocks/discrete_transformer.cpp
    blocks/pattern_classifier.cpp
    blocks/pattern_classifier_dynamic.cpp
    blocks/pattern_pooler.cpp
    blocks/persistence_transformer.cpp
    blocks/scalar_transformer.cpp
    blocks/sequence_learner.cpp
)

add_library(bbcore STATIC ${SOURCE_FILES})
