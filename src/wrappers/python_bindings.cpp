#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

//#include "utils.hpp"
#include "bitarray.hpp"
#include "page.hpp"
#include "coincidence_set.hpp"
#include "blank_block.hpp"
#include "scalar_encoder.hpp"
#include "symbols_encoder.hpp"
#include "persistence_encoder.hpp"
#include "pattern_classifier.hpp"
#include "pattern_pooler.hpp"
#include "sequence_learner.hpp"

// shorthand the pybind11 namespace
namespace py = pybind11;

// allows us to use shorthand for adding keywords to python function definitions
using namespace py::literals;

PYBIND11_MODULE(bb_backend, m) {
    m.doc() = R"pbdoc(
        BrainBlocks Python Module
        -----------------------
        -------------------------

        .. currentmodule:: bb_backend

        .. autosummary::
           :toctree: _generate

           BlankBlock
           ScalarEncoder
           SymbolsEncoder
           PersistenceEncoder
           PatternPooler
           PatternClassifier
           SequenceLearner

    )pbdoc";

    // seed
    //m.def("seed", &seed);

    // BitArray
    py::class_<BitArray>(m, "BitArray")
        .def("print_info", &BitArray::print_info, "Print BitArray information.")
        .def("set_bits", &BitArray::set_bits, "Set an array of bits.", "bits"_a)
        .def("set_acts", &BitArray::set_acts, "Set an array of acts.", "acts"_a)
        .def("get_bits", &BitArray::get_bits, "Get an array of bits.")
        .def("get_acts", &BitArray::get_acts, "Get an array of acts.")
        .def_property_readonly("num_bits", &BitArray::get_num_bits, "Get number of bits in BitArray.")
        .def_property_readonly("num_words", &BitArray::get_num_words, "Get number of 32-bit words in BitArray.");

    // Page
    py::class_<Page>(m, "Page")
        .def("print_info", &Page::print_info, "Print Page information.")
        .def("add_child", &Page::add_child, "Add a child Page to this parent Page.", "child"_a)
        //.def("get_child", &Page::get_child, "Get a particular child page by index.", "child_index"_a) // TODO
        .def("__getitem__", &Page::operator[], "Get a particular BitArray by the time index.", "t"_a, py::return_value_policy::reference_internal) // TODO: verify if return_valy_policy is truly needed
        .def_property_readonly("num_bitarrays", &Page::get_num_bitarrays, "Get number of BitArrays.")
        .def_property_readonly("num_children", &Page::get_num_children, "Get number of child Pages.");

    // CoincidenceSet
    py::class_<CoincidenceSet>(m, "CoincidenceSet")
        .def("get_addr", &CoincidenceSet::get_addr, "Get a particular address.", "idx"_a)
        .def("get_perm", &CoincidenceSet::get_perm, "Get a particular permanence.", "idx"_a)
        .def("get_addrs", &CoincidenceSet::get_addrs, "Get array of addresses.")
        .def("get_perms", &CoincidenceSet::get_perms, "Get array of permanences.");
        //.def("get_bits", &CoincidenceSet::get_bits, "Get array of bits representing receptor connections.") // TODO
        //.def("get_acts", &CoincidenceSet::get_acts, "Get array of acts representing receptor connections.") // TODO

    // Blank Block
    py::class_<BlankBlock>(m, "BlankBlock")
        .def(py::init<const uint32_t>(), "Construct BlankBlock.", "num_s"_a)
        .def("clear_states", &BlankBlock::clear_states, "Clear states in the Block.")
        .def_property_readonly("output", &BlankBlock::get_output, "Get output Page object.");

    // Scalar Encoder
    py::class_<ScalarEncoder>(m, "ScalarEncoder")
        .def(py::init<const double, const double, const uint32_t, const uint32_t>(), "Construct ScalarEncoder.", "min_val"_a, "max_val"_a, "num_s"_a, "num_as"_a)
        .def("initialize", &ScalarEncoder::initialize, "Initialize Block.")
        .def("clear_states", &ScalarEncoder::clear_states, "Clear states in the Block.")
        .def("compute", &ScalarEncoder::compute, "Compute Block.", "value"_a)
        .def_property_readonly("output", &ScalarEncoder::get_output, "Get output Page object.");

    // Symbols Encoder
    py::class_<SymbolsEncoder>(m, "SymbolsEncoder")
        .def(py::init<const uint32_t, const uint32_t>(), "Construct SymbolEncoder.", "max_symbols"_a, "num_s"_a)
        .def("initialize", &SymbolsEncoder::initialize, "Initialize Block.")
        .def("clear_states", &SymbolsEncoder::clear_states, "Clear states in the Block.")
        .def("compute", &SymbolsEncoder::compute, "Compute Block.", "value"_a)
        .def_property_readonly("output", &SymbolsEncoder::get_output, "Get output Page object.");

    // Persistence Encoder
    py::class_<PersistenceEncoder>(m, "PersistenceEncoder")
        .def(py::init<const double, const double, const uint32_t, const uint32_t, const uint32_t>(), "Construct PersistenceEncoder.", "min_val"_a, "max_val"_a, "num_s"_a, "num_as"_a, "max_steps"_a)
        .def("initialize", &PersistenceEncoder::initialize, "Initialize Block.")
        .def("clear_states", &PersistenceEncoder::clear_states, "Clear states in the Block.")
        .def("compute", &PersistenceEncoder::compute, "Compute Block.", "value"_a)
        .def_property_readonly("output", &PersistenceEncoder::get_output, "Get output Page object.");

    // Pattern Classifier
    py::class_<PatternClassifier>(m, "PatternClassifier")
        .def(py::init<const std::vector<uint32_t>, const uint32_t, const uint32_t, const uint8_t, const uint8_t, const uint8_t, const double, const double, const double>(), "Construct PatternClassifier.", "labels"_a, "num_s"_a, "num_as"_a, "perm_thr"_a, "perm_inc"_a, "perm_dec"_a, "pct_pool"_a, "pct_conn"_a, "pct_learn"_a)
        .def("initialize", &PatternClassifier::initialize, "Initialize the Block.")
        .def("save", &PatternClassifier::save, "Save Block memories.", "file_str"_a)
        .def("load", &PatternClassifier::load, "Load Block memories.", "file_str"_a)
        .def("clear_states", &PatternClassifier::clear_states, "Clear states in the Block.")
        .def("compute", &PatternClassifier::compute, "Compute Block.", "label"_a, "learn"_a)
        .def("get_labels", &PatternClassifier::get_labels, "Get Block labels.")
        .def("get_probabilities", &PatternClassifier::get_probabilities, "Get Block labels.")
        .def("output_coincidence_set", &PatternClassifier::get_output_coincidence_set, "Get a particular output CoincidenceSet object.", "d"_a, py::return_value_policy::reference_internal)
        .def_property_readonly("input", &PatternClassifier::get_input, "Get input Page object.")
        .def_property_readonly("output", &PatternClassifier::get_output, "Get output Page object.");

    // Pattern Pooler
    py::class_<PatternPooler>(m, "PatternPooler")
        .def(py::init<const uint32_t, const uint32_t, const uint8_t, const uint8_t, const uint8_t, const double, const double, const double>(), "Construct PatternPooler.", "num_s"_a, "num_as"_a, "perm_thr"_a, "perm_inc"_a, "perm_dec"_a, "pct_pool"_a, "pct_conn"_a, "pct_learn"_a)
        .def("initialize", &PatternPooler::initialize, "Initialize the Block.")
        .def("save", &PatternPooler::save, "Save Block memories.", "file_str"_a)
        .def("load", &PatternPooler::load, "Load Block memories.", "file_str"_a)
        .def("clear_states", &PatternPooler::clear_states, "Clear states in the Block.")
        .def("compute", &PatternPooler::compute, "Compute Block.", "learn"_a)
        .def("output_coincidence_set", &PatternPooler::get_output_coincidence_set, "Get a particular output CoincidenceSet object.", "d"_a, py::return_value_policy::reference_internal)
        .def_property_readonly("input", &PatternPooler::get_input, "Get input Page object.")
        .def_property_readonly("output", &PatternPooler::get_output, "Get output Page object.");

    // Sequence Learner
    py::class_<SequenceLearner>(m, "SequenceLearner")
        .def(py::init<const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint8_t, const uint8_t, const uint8_t>(), "Construct SequenceLearner", "num_spc"_a, "num_dps"_a, "num_rpd"_a, "d_thresh"_a, "perm_thr"_a, "perm_inc"_a, "perm_dec"_a)
        .def("initialize", &SequenceLearner::initialize, "Initialize the Block.")
        .def("save", &SequenceLearner::save, "Save Block memories.", "file_str"_a)
        .def("load", &SequenceLearner::load, "Load Block memories.", "file_str"_a)
        .def("clear_states", &SequenceLearner::clear_states, "Clear states in the Block.")
        .def("compute", &SequenceLearner::compute, "Compute Block.", "learn"_a)
        .def("get_score", &SequenceLearner::get_score, "Get abnormality score.")
        //.def("get_historical_count", &SequenceLearner::get_historical_count, "Get number of historical statelets")
		//.def("get_coincidence_set_count", &SequenceLearner::get_coincidence_set_count, "Get number of used coincidence sets")
		//.def("get_historical_statelets", &SequenceLearner::get_historical_statelets, "Get historical statelets")
		//.def("get_num_coincidence_sets_per_statelet", &SequenceLearner::get_num_coincidence_sets_per_statelet, "Get number of coincidence sets per statelet")
        .def("hidden_coincidence_set", &SequenceLearner::get_hidden_coincidence_set, "Get a particular hidden CoincidenceSet object.", "d"_a, py::return_value_policy::reference_internal)
        .def("output_coincidence_set", &SequenceLearner::get_output_coincidence_set, "Get a particular output CoincidenceSet object.", "d"_a, py::return_value_policy::reference_internal)
        .def_property_readonly("input", &SequenceLearner::get_input, "Get input Page object.")
        .def_property_readonly("hidden", &SequenceLearner::get_hidden, "Get hidden Page object.")
        .def_property_readonly("output", &SequenceLearner::get_output, "Get output Page object.");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}