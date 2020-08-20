#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "wrappers/class_interfaces.hpp"

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
    m.def("seed", &seed);

    // Page
    py::class_<PageClass>(m, "Page")
        .def("add_child", &PageClass::add_child, "Add a child page to this parent page", "child"_a)
        .def("get_child", &PageClass::get_child, "Get a particular child page by index", "child_index"_a)
        .def("get_bits", &PageClass::get_bits, "Get a particular array of bits by the time index", "t"_a)
        .def("set_bits", &PageClass::set_bits, "Set a particular array of bits by the time index", "t"_a, "bits"_a)
        .def("get_acts", &PageClass::get_acts, "Get a particular array of acts by the time index", "t"_a)
        .def("set_acts", &PageClass::set_acts, "Set a particular array of acts by the time index", "t"_a, "acts"_a)
        .def_property_readonly("num_children", &PageClass::num_children, "Get number of children pages")
        .def_property_readonly("num_history", &PageClass::num_history, "Get number of histories");

    // CoincidenceSet
    py::class_<CoincidenceSetClass>(m, "CoincidenceSet")
        .def("get_addrs", &CoincidenceSetClass::get_addrs, "Get addresses")
        .def("get_addr", &CoincidenceSetClass::get_addr, "Get a particular address", "d"_a)
        .def("get_perms", &CoincidenceSetClass::get_perms, "Get permanences")
        .def("get_perm", &CoincidenceSetClass::get_perm, "Get a particular permanence", "d"_a)
        .def("get_bits", &CoincidenceSetClass::get_bits, "Get array of bits representing receptor connections")
        .def("get_acts", &CoincidenceSetClass::get_acts, "Get array of acts representing receptor connections");

    // Blank Block
    py::class_<BlankBlockClass>(m, "BlankBlock")
        .def(py::init<const uint32_t>(), "Blank Block Creation", "num_s"_a)
        .def_property_readonly("output", &BlankBlockClass::get_output, "Get output Page object");

    // Scalar Encoder
    py::class_<ScalarEncoderClass>(m, "ScalarEncoder")
        .def(py::init<const double, const double, const uint32_t, const uint32_t>(),
            "Construct ScalarEncoder", "min_val"_a, "max_val"_a, "num_s"_a, "num_as"_a)
        .def("compute", &ScalarEncoderClass::compute, "Compute block", "value"_a)
        .def_property_readonly("output", &ScalarEncoderClass::get_output, "Get output Page object");

    // Symbols Encoder
    py::class_<SymbolsEncoderClass>(m, "SymbolsEncoder")
        .def(py::init<const uint32_t, const uint32_t>(),
            "Construct SymbolEncoder", "max_symbols"_a, "num_s"_a)
        .def("compute", &SymbolsEncoderClass::compute, "Compute block", "value"_a)
        .def("get_symbols", &SymbolsEncoderClass::get_symbols, "Get symbols")
        .def_property_readonly("output", &SymbolsEncoderClass::get_output, "Get output Page object");

    // Persistence Encoder
    py::class_<PersistenceEncoderClass>(m, "PersistenceEncoder")
        .def(py::init<const double, const double, const uint32_t, const uint32_t, const uint32_t>(),
            "Construct PersistenceEncoder", "min_val"_a, "max_val"_a, "num_s"_a, "num_as"_a, "max_steps"_a)
        .def("reset", &PersistenceEncoderClass::reset, "Reset persistence")
        .def("compute", &PersistenceEncoderClass::compute, "Compute block", "value"_a)
        .def_property_readonly("output", &PersistenceEncoderClass::get_output, "Get output Page object");

    // Pattern Classifier
    py::class_<PatternClassifierClass>(m, "PatternClassifier")
        .def(py::init<const std::vector<uint32_t>, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const double, const double, const double>(),
            "Construct PatternClassifier", "labels"_a, "num_l"_a, "num_s"_a, "num_as"_a, "perm_thr"_a, "perm_inc"_a, "perm_dec"_a, "pct_pool"_a, "pct_conn"_a, "pct_learn"_a)
        .def("initialize", &PatternClassifierClass::initialize, "Initialize block")
        .def("save", &PatternClassifierClass::save, "Save block")
        .def("load", &PatternClassifierClass::load, "Load block")
        .def("compute", &PatternClassifierClass::compute, "Compute block", "label"_a, "learn"_a)
        .def("get_probabilities", &PatternClassifierClass::get_probabilities, "Get label probabilities")
        .def("coincidence_set", &PatternClassifierClass::get_coincidence_set, "Get a particular CoincidenceSet object", "d"_a)
        .def_property_readonly("input", &PatternClassifierClass::get_input, "Get input Page object")
        .def_property_readonly("output", &PatternClassifierClass::get_output, "Get output Page object");

    // Pattern Pooler
    py::class_<PatternPoolerClass>(m, "PatternPooler")
        .def(py::init<const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const double, const double, const double>(),
            "Construct PatternPooler", "num_s"_a, "num_as"_a, "perm_thr"_a, "perm_inc"_a, "perm_dec"_a, "pct_pool"_a, "pct_conn"_a, "pct_learn"_a)
        .def("initialize", &PatternPoolerClass::initialize, "Initialize block")
        .def("save", &PatternPoolerClass::save, "Save block")
        .def("load", &PatternPoolerClass::load, "Load block")
        .def("compute", &PatternPoolerClass::compute, "Compute block", "learn"_a)
        .def("coincidence_set", &PatternPoolerClass::get_coincidence_set, "Get a particular CoincidenceSet object", "d"_a)
        .def_property_readonly("input", &PatternPoolerClass::get_input, "Get input Page object")
        .def_property_readonly("output", &PatternPoolerClass::get_output, "Get output Page object");

    // Sequence Learner
    py::class_<SequenceLearnerClass>(m, "SequenceLearner")
        .def(py::init<const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t>(), 
            "Construct SequenceLearner", "num_spc"_a, "num_dps"_a, "num_rpd"_a, "d_thresh"_a, "perm_thr"_a, "perm_inc"_a, "perm_dec"_a)
        .def("initialize", &SequenceLearnerClass::initialize, "Initialize block")
        .def("save", &SequenceLearnerClass::save, "Save block")
        .def("load", &SequenceLearnerClass::load, "Load block")
        .def("compute", &SequenceLearnerClass::compute, "Compute block")
        .def("get_score", &SequenceLearnerClass::get_score, "Get abnormality score")
        .def("get_historical_count", &SequenceLearnerClass::get_historical_count, "Get number of historical statelets")
		.def("get_coincidence_set_count", &SequenceLearnerClass::get_coincidence_set_count, "Get number of used coincidence sets")
		.def("get_historical_statelets", &SequenceLearnerClass::get_historical_statelets, "Get historical statelets")
		.def("get_num_coincidence_sets_per_statelet", &SequenceLearnerClass::get_num_coincidence_sets_per_statelet, "Get number of coincidence sets per statelet")
        .def("get_hidden_coincidence_set", &SequenceLearnerClass::get_hidden_coincidence_set, "Get a particular hidden CoincidenceSet object", "d"_a)
        .def("get_output_coincidence_set", &SequenceLearnerClass::get_output_coincidence_set, "Get a particular output CoincidenceSet object", "d"_a)
        .def_property_readonly("input", &SequenceLearnerClass::get_input, "Get input Page object")
        .def_property_readonly("hidden", &SequenceLearnerClass::get_hidden, "Get hidden Page object")
        .def_property_readonly("output", &SequenceLearnerClass::get_output, "Get output Page object");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}