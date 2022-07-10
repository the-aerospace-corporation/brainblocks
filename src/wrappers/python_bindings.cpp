// =============================================================================
// python_bindings.cpp
// =============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bitarray.hpp"
#include "block.hpp"
#include "block_input.hpp"
#include "block_memory.hpp"
#include "block_output.hpp"

#include "blocks/blank_block.hpp"
#include "blocks/context_learner.hpp"
#include "blocks/discrete_transformer.hpp"
#include "blocks/pattern_classifier.hpp"
#include "blocks/pattern_classifier_dynamic.hpp"
#include "blocks/pattern_pooler.hpp"
#include "blocks/persistence_transformer.hpp"
#include "blocks/scalar_transformer.hpp"
#include "blocks/sequence_learner.hpp"

// Shorthand the pybind11 namespace
namespace py = pybind11;

// Allows use of shorthand for adding keywords to python function definitions
using namespace py::literals;

using namespace BrainBlocks;

PYBIND11_MODULE(bb_backend, m) {
    m.doc() = R"pbdoc(
        BrainBlocks Python Module
        -------------------------

        .. currentmodule:: bb_backend

        .. autosummary::
           :toctree: _generate

           BlankBlock
           ScalarTransformer
           SymbolsEncoder
           PersistenceTransformer
           PatternPooler
           PatternClassifier
           SequenceLearner

    )pbdoc";

    // seed
    //m.def("seed", &seed);

    // =========================================================================
    // BitArray
    // =========================================================================
    py::class_<BitArray>(m, "BitArray")

        .def("set_bits", &BitArray::set_bits,
             "Set the BitArray from a vector of bits", "bits"_a)

        .def("set_acts", &BitArray::set_acts,
             "Set the BitArray from a vector of acts", "acts"_a)

        .def("get_bits", &BitArray::get_bits,
             "Returns a vector of bits from the BitArray")

        .def("get_acts", &BitArray::get_acts,
             "Returns a vector of acts from the BitArray")

        .def_property_readonly("num_bits", &BitArray::num_bits,
                               "Returns the number of bits in BitArray")

        .def_property_readonly("num_words", &BitArray::num_words,
                               "Returns the number of words in BitArray");

    // =========================================================================
    // Block
    // =========================================================================
    py::class_<Block>(m, "Block")

        .def("init", &Block::init,
             "Initializes BlockMemories based on BlockInput parameters")

        .def("save", &Block::save, "file"_a, "Save block memories")

        .def("load", &Block::load, "file"_a, "Load Block memories")

        .def("clear", &Block::clear,
             "Clears BlockInput, BlockMemory, and BlockOutput states")

        .def("step", &Block::step,
             "Updates BlockOutput history current index")

        .def("pull", &Block::pull,
             "Updates BlockInput state(s) from child BlockOutput histories")

        .def("push", &Block::push,
             "Updates child BlockOutput state(s) from BlockInput state(s)")

        .def("encode", &Block::encode,
             "Converts BlockInput state(s) into BlockOutput state(s)")

        .def("decode", &Block::decode,
             "Converts BlockOutput state(s) into BlockInput state(s)")

        .def("learn", &Block::learn, "Updates BlockMemories")

        .def("store", &Block::store,
             "Copy BlockOutput state into current index of BlockOutput history")

        .def("memory_usage", &Block::memory_usage,
             "Returns an estimate of the number of bytes used by the block")

        .def("feedforward", &Block::feedforward, "learn_flag"_a=false,
             "Performs all functions required to produce output from intput")

        .def("feedback", &Block::feedback,
             "Performs all funtions required to produce input from output");

    // =========================================================================
    // BlockInput
    // =========================================================================
    py::class_<BlockInput>(m, "BlockInput")

        .def("add_child", &BlockInput::add_child,
             "Connects a BlockOutput at a prior time step to the BlockInput",
             "src"_a, "src_t"_a)

        .def_property_readonly("num_children", &BlockInput::num_children,
                               "Returns number of children in BlockInput")

        .def_readonly("state", &BlockInput::state,
                      "Returns state BitArray object");

    // =========================================================================
    // BlockMemory
    // =========================================================================
    py::class_<BlockMemory>(m, "BlockMemory")

        .def("addrs", &BlockMemory::addrs,
             "Returns a particular dendrite's receptor addresses", "d"_a)

        .def("perms", &BlockMemory::perms,
             "Returns a particular dendrite's receptor permanences", "d"_a)

        .def("conns", &BlockMemory::conns,
             "Returns a particular dendrite's receptor connections", "d"_a)

        .def_property_readonly("num_dendrites", &BlockMemory::num_dendrites,
                               "Returns number of dendrites");


    // =========================================================================
    // BlockOutput
    // =========================================================================
    py::class_<BlockOutput>(m, "BlockOutput")

        .def("get_bitarray", &BlockOutput::get_bitarray,
             "Returns a bitarray based on the inputted time step", "t"_a)

        .def_property_readonly("num_t", &BlockOutput::num_t,
                               "Returns number of time steps in BlockOutput")

        .def_readonly("state", &BlockOutput::state,
                      "Returns state BitArray object");

    // =========================================================================
    // BlankBlock
    // =========================================================================
    py::class_<BlankBlock, Block>(m, "BlankBlock")

        .def(py::init<
	        const uint32_t,
            const uint32_t,
	        const uint32_t>(),
        "num_s"_a,
	    "num_t"_a,
        "seed"_a=0,
	    "Constructs a BlankBlock")

        .def_readonly("output", &BlankBlock::output,
                      "Returns output BlockOutput object");

    // =========================================================================
    // ContextLearner
    // =========================================================================
    py::class_<ContextLearner, Block>(m, "ContextLearner")

        .def(py::init<
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint8_t,
            const uint8_t,
            const uint8_t,
            const uint32_t,
            const uint32_t>(),
        "num_c"_a,
        "num_spc"_a,
        "num_dps"_a,
        "num_rpd"_a,
        "d_thresh"_a,
        "perm_thr"_a,
        "perm_inc"_a,
        "perm_dec"_a,
        "num_t"_a=2,
        "seed"_a=0,
        "Constructs a ContextLearner")

        .def("get_anomaly_score", &ContextLearner::get_anomaly_score,
             "Returns anomaly score")

        .def_readonly("input", &ContextLearner::input,
                      "Returns input BlockInput object")

        .def_readonly("context", &ContextLearner::context,
                      "Returns context BlockInput object")

        .def_readonly("output", &ContextLearner::output,
                      "Returns output BlockOutput object")

        .def_readonly("memory", &ContextLearner::memory,
                      "Returns memory BlockMemory object");

    // =========================================================================
    // DiscreteTransformer
    // =========================================================================
    py::class_<DiscreteTransformer, Block>(m, "DiscreteTransformer")

        .def(py::init<
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t>(),
        "num_v"_a,
        "num_s"_a,
        "num_t"_a=2,
        "seed"_a=0,
        "Constructs a DiscreteTransformer")

        .def("set_value", &DiscreteTransformer::set_value, "value"_a,
             "Sets value")

        .def("get_value", &DiscreteTransformer::get_value,
             "Returns value")

        .def_readonly("output", &DiscreteTransformer::output,
                      "Returns output BlockOutput object");

    // =========================================================================
    // PatternClassifier
    // =========================================================================
    py::class_<PatternClassifier, Block>(m, "PatternClassifier")

        .def(py::init<
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint8_t,
            const uint8_t,
            const uint8_t,
            const double,
            const double,
            const double,
            const uint32_t,
            const uint32_t>(),
        "num_l"_a,
        "num_s"_a,
        "num_as"_a,
        "perm_thr"_a,
        "perm_inc"_a,
        "perm_dec"_a,
        "pct_pool"_a,
        "pct_conn"_a,
        "pct_learn"_a,
        "num_t"_a=2,
        "seed"_a=0,
        "Constructs a PatternClassifier")

        .def("set_label", &PatternClassifier::set_label, "label"_a,
             "Sets label")

        .def("get_labels", &PatternClassifier::get_labels,
             "Returns array of stored labels")

        .def("get_probabilities", &PatternClassifier::get_probabilities,
             "Returns array of probability scores for each stored label")

        .def_readonly("input", &PatternClassifier::input,
                      "Returns input BlockInput object")

        .def_readonly("output", &PatternClassifier::output,
                      "Returns output BlockOutput object")

        .def_readonly("memory", &PatternClassifier::memory,
                      "Returns memory BlockMemory object");

    // =========================================================================
    // PatternClassifierDynamic
    // =========================================================================
    py::class_<PatternClassifierDynamic, Block>(m, "PatternClassifierDynamic")

        .def(py::init<
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint8_t,
            const uint8_t,
            const uint8_t,
            const double,
            const double,
            const double,
            const uint32_t,
            const uint32_t>(),
        "num_s"_a,
        "num_as"_a,
        "num_spl"_a,
        "perm_thr"_a,
        "perm_inc"_a,
        "perm_dec"_a,
        "pct_pool"_a,
        "pct_conn"_a,
        "pct_learn"_a,
        "num_t"_a=2,
        "seed"_a=0,
        "Constructs a PatternClassifierDynamic")

        .def("set_label", &PatternClassifierDynamic::set_label, "label"_a,
             "Sets label")

        .def("get_anomaly_score", &PatternClassifierDynamic::get_anomaly_score,
             "Returns anomaly score")

        .def("get_labels", &PatternClassifierDynamic::get_labels,
             "Returns array of stored labels")

        .def("get_probabilities", &PatternClassifierDynamic::get_probabilities,
             "Returns array of probability scores for each stored label")

        .def_readonly("input", &PatternClassifierDynamic::input,
                      "Returns input BlockInput object")

        .def_readonly("output", &PatternClassifierDynamic::output,
                      "Returns output BlockOutput object")

        .def_readonly("memory", &PatternClassifierDynamic::memory,
                      "Returns memory BlockMemory object");

    // =========================================================================
    // PatternPooler
    // =========================================================================
    py::class_<PatternPooler, Block>(m, "PatternPooler")

        .def(py::init<
            const uint32_t,
            const uint32_t,
            const uint8_t,
            const uint8_t,
            const uint8_t,
            const double,
            const double,
            const double,
            const uint32_t,
            const bool,
            const uint32_t>(),
        "num_s"_a,
        "num_as"_a,
        "perm_thr"_a,
        "perm_inc"_a,
        "perm_dec"_a,
        "pct_pool"_a,
        "pct_conn"_a,
        "pct_learn"_a,
        "num_t"_a=2,
        "always_update"_a=false,
        "seed"_a,
        "Constructs a PatternPooler")

        .def_readonly("input", &PatternPooler::input,
                      "Returns input BlockInput object")

        .def_readonly("output", &PatternPooler::output,
                      "Returns output BlockOutput object")

        .def_readonly("memory", &PatternPooler::memory,
                      "Returns memory BlockMemory object");

    // =========================================================================
    // PersistenceTransformer
    // =========================================================================
    py::class_<PersistenceTransformer, Block>(m, "PersistenceTransformer")

        .def(py::init<
            const double,
            const double,
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t>(),
        "min_val"_a,
        "max_val"_a,
        "num_s"_a,
        "num_as"_a,
        "max_step"_a,
        "num_t"_a=2,
        "seed"_a=0,
        "Constructs a PersistenceTransformer")

        .def("set_value", &PersistenceTransformer::set_value, "value"_a,
             "Sets value")

        .def("get_value", &PersistenceTransformer::get_value,
             "Returns value")

        .def_readonly("output", &PersistenceTransformer::output,
                      "Returns output BlockOutput object");

    // =========================================================================
    // ScalarTransformer
    // =========================================================================
    py::class_<ScalarTransformer, Block>(m, "ScalarTransformer")

        .def(py::init<
            const double,
            const double,
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t>(),
        "min_val"_a,
        "max_val"_a,
        "num_s"_a,
        "num_as"_a,
        "num_t"_a=2,
        "seed"_a=0,
        "Constructs a ScalarTransformer")

        .def("set_value", &ScalarTransformer::set_value, "value"_a,
             "Sets value")

        .def("get_value", &ScalarTransformer::get_value,
             "Returns value")

        .def_readonly("output", &ScalarTransformer::output,
                      "Returns output BlockOutput object");

    // =========================================================================
    // SequenceLearner
    // =========================================================================
    py::class_<SequenceLearner, Block>(m, "SequenceLearner")

        .def(py::init<
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint32_t,
            const uint8_t,
            const uint8_t,
            const uint8_t,
            const uint32_t,
            const bool,
            const uint32_t>(),
        "num_c"_a,
        "num_spc"_a,
        "num_dps"_a,
        "num_rpd"_a,
        "d_thresh"_a,
        "perm_thr"_a,
        "perm_inc"_a,
        "perm_dec"_a,
        "num_t"_a=2,
        "always_update"_a=false,
        "seed"_a=0,
        "Constructs a SequenceLearner")

        .def("get_anomaly_score", &SequenceLearner::get_anomaly_score,
             "Returns anomaly score")

        .def("get_historical_count", &SequenceLearner::get_historical_count,
             "Get number of historical statelets")

        .def_readonly("input", &SequenceLearner::input,
                      "Returns input BlockInput object")

        .def_readonly("context", &SequenceLearner::context,
                      "Returns context BlockInput object")

        .def_readonly("output", &SequenceLearner::output,
                      "Returns output BlockOutput object")

        .def_readonly("memory", &SequenceLearner::memory,
                      "Returns memory BlockMemory object");



#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
