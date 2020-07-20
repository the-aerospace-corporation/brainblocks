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
        .def("get_perm", &CoincidenceSetClass::get_perm, "Get a particular permanence", "d"_a);

    // Blank Block
    py::class_<BlankBlockClass>(m, "BlankBlock")
        .def(py::init<const uint32_t>(), "Blank Block Creation", "num_s"_a)
        .def_property_readonly("output", &BlankBlockClass::get_output, "Get output Page object")
        .def("clear", &BlankBlockClass::clear, "Clear block");

    // Scalar Encoder
    py::class_<ScalarEncoderClass>(m, "ScalarEncoder")
        .def(py::init<const double, const double, const uint32_t, const uint32_t>(),
            "Construct ScalarEncoder", "min_val"_a, "max_val"_a, "num_s"_a, "num_as"_a)
        .def("clear", &ScalarEncoderClass::clear, "Clear block")
        .def("compute", &ScalarEncoderClass::compute, "Compute block", "value"_a)
        .def_property_readonly("output", &ScalarEncoderClass::get_output, "Get output Page object");

    // Symbols Encoder
    py::class_<SymbolsEncoderClass>(m, "SymbolsEncoder")
        .def(py::init<const uint32_t, const uint32_t>(),
            "Construct SymbolEncoder", "max_symbols"_a, "num_s"_a)
        .def("clear", &SymbolsEncoderClass::clear, "Clear block")
        .def("compute", &SymbolsEncoderClass::compute, "Compute block", "value"_a)
        .def("get_symbols", &SymbolsEncoderClass::get_symbols, "Get symbols")
        .def_property_readonly("output", &SymbolsEncoderClass::get_output, "Get output Page object");

    // Persistence Encoder
    py::class_<PersistenceEncoderClass>(m, "PersistenceEncoder")
        .def(py::init<const double, const double, const uint32_t, const uint32_t, const uint32_t>(),
            "Construct PersistenceEncoder", "min_val"_a, "max_val"_a, "num_s"_a, "num_as"_a, "max_steps"_a)
        .def("reset", &PersistenceEncoderClass::reset, "Reset persistence")
        .def("clear", &PersistenceEncoderClass::clear, "Clear block")
        .def("compute", &PersistenceEncoderClass::compute, "Compute block", "value"_a)
        .def_property_readonly("output", &PersistenceEncoderClass::get_output, "Get output Page object");

    // Pattern Classifier
    py::class_<PatternClassifierClass>(m, "PatternClassifier")
        .def(py::init<const std::vector<uint32_t>, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const double, const double, const double>(),
            "Construct PatternClassifier", "labels"_a, "num_l"_a, "num_s"_a, "num_as"_a, "perm_thr"_a, "perm_inc"_a, "perm_dec"_a, "pct_pool"_a, "pct_conn"_a, "pct_learn"_a)
        .def("initialize", &PatternClassifierClass::initialize, "Initialize block")
        .def("save", &PatternClassifierClass::save, "Save block")
        .def("load", &PatternClassifierClass::load, "Load block")
        .def("clear", &PatternClassifierClass::clear, "Clear block")
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
        .def("clear", &PatternPoolerClass::clear, "Clear block")
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
        .def("clear", &SequenceLearnerClass::clear, "Clear block")
        .def("compute", &SequenceLearnerClass::compute, "Compute block")
        .def("get_score", &SequenceLearnerClass::get_score, "Get abnormality score")
        .def("coincidence_set", &SequenceLearnerClass::get_coincidence_set, "Get a particular CoincidenceSet object", "d"_a)
        .def_property_readonly("input", &SequenceLearnerClass::get_input, "Get input Page object")
        .def_property_readonly("output", &SequenceLearnerClass::get_output, "Get output Page object");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}