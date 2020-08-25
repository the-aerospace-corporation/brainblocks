extern "C" {
    #include "utils.h"
    #include "page.h"
    #include "blank_block.h"
    #include "scalar_encoder.h"
    #include "symbols_encoder.h"
    #include "persistence_encoder.h"
    #include "pattern_classifier.h"
    #include "pattern_pooler.h"
    #include "sequence_learner.h"
}

// TODO: convert pointers to smart pointers
#include <string>
#include <vector>
#include <stdexcept>


void seed(const uint32_t seed) {
    utils_seed(seed);
}


// =============================================================================
// PageClass
// =============================================================================
class PageClass {
    public:
        // Construct Page
        PageClass(Page* ptr) {page = ptr; };

        // Deconstruct Page
        //~PageClass() {};

        // Add a child page to this parent page
        void add_child(PageClass* child) {
            page_add_child(page, child->get_ptr());
        };

        // Get a particular child page by index
        PageClass* get_child(const uint32_t child_index) {
            return new PageClass(page->children[child_index]);
        };

        // Get a particular array of bits by the time index
        std::vector<uint8_t> get_bits(const uint32_t t) {
            struct BitArray* ba = page_get_bitarray(page, t);
            uint32_t num_bits = ba->num_bits;
            std::vector<uint8_t> bits;
            bits.reserve(num_bits);

            for (uint32_t i = 0; i < num_bits; i++) {
                bits.push_back(page_get_bit(page, t, i));
            }

            return bits;
        };

        // Set a particular array of bits by the time index
        void set_bits(const uint32_t t, std::vector<bool> new_bits) {
            struct BitArray* ba = page_get_bitarray(page, t);
            uint32_t num_bits = ba->num_bits;

            if (new_bits.size() > num_bits) {
                throw std::range_error("Error: Number of bits exceeds the maximum allocation for this BitArray");
            }

            page_clear_bits(page, t);

            for (uint32_t i = 0; i < num_bits; i++) {
                if (new_bits[i] != 0) {
                    page_set_bit(page, t, i);
                }
            }
        };

        // Get a particular array of acts by the time index
        std::vector<uint32_t> get_acts(const uint32_t t) {
            struct ActArray* aa = page_get_actarray(page, t);
            uint32_t num_acts = aa->num_acts;
            std::vector<uint32_t> acts;
            acts.reserve(num_acts);

            for (uint32_t i = 0; i < num_acts; i++) {
                acts.push_back(aa->acts[i]);
            }

            return acts;
        };

        // Set a particular array of acts by the time index
        void set_acts(const uint32_t t, std::vector<uint32_t> new_acts) {
            struct BitArray* ba = page_get_bitarray(page, t);
            uint32_t num_bits = ba->num_bits;
            uint32_t num_acts = (uint32_t)new_acts.size();

            if (num_acts > num_bits) {
                throw std::range_error("Number of acts exceeds the maximum allocation for this BitArray");
            }

            page_clear_bits(page, t);

            for (uint32_t i = 0; i < num_acts; i++) {
                page_set_bit(page, t, new_acts[i]);
            }
        };

        // Get number of children pages
        size_t num_children() const { 
            return page->num_children;
        };

        // Get number of histories
        size_t num_history() const {
            return page->num_history;
        };

    private:
        Page* get_ptr() { return page; };
        struct Page* page;
};


// =============================================================================
// CoincidenceSetClass
// =============================================================================
class CoincidenceSetClass {
    public:
        // Construct CoincidenceSet
        CoincidenceSetClass(CoincidenceSet* ptr) {d = ptr; };

        // Deconstruct CoincidenceSet
        //~CoincidenceSetClass() {};

        // Get addresses
        std::vector<uint32_t> get_addrs() {
            std::vector<uint32_t> addrs;
            addrs.reserve(d->num_r);

            for (uint32_t r = 0; r < d->num_r; r++) {
                addrs.push_back(d->addrs[r]);
            }

            return addrs;
        }

        // Get a particular address
        uint32_t get_addr(const uint32_t r) {
            if (r > d->num_r) {
                throw std::range_error("Error: Index out of range in get_addrs()");
            }

            return d->addrs[r];
        }

        // Get permanences
        std::vector<uint32_t> get_perms() {
            std::vector<uint32_t> perms;
            perms.reserve(d->num_r);

            for (uint32_t r = 0; r < d->num_r; r++) {
                perms.push_back(d->perms[r]);
            }

            return perms;
        }

        // Get a particular permanence
        uint32_t get_perm(const uint32_t r) {
            if (r > d->num_r) {
                throw std::range_error("Error: Index out of range in get_perms()");
            }

            return d->perms[r];
        }

        // Get array of bits representing receptor connections
        std::vector<uint8_t> get_bits() {
            struct BitArray* ba = coincidence_set_get_connections(d);
            uint32_t num_bits = ba->num_bits;
            std::vector<uint8_t> bits;
            bits.reserve(num_bits);

            for (uint32_t i = 0; i < num_bits; i++) {
                bits.push_back(bitarray_get_bit(ba, i));
            }

            return bits;
        };
        
        // Get array of acts representing receptor connections
        std::vector<uint32_t> get_acts() {
            struct BitArray* ba = coincidence_set_get_connections(d);
            struct ActArray* aa = bitarray_get_actarray(ba);
            uint32_t num_acts = aa->num_acts;
            std::vector<uint32_t> acts;
            acts.reserve(num_acts);

            for (uint32_t i = 0; i < num_acts; i++) {
                acts.push_back(aa->acts[i]);
            }

            return acts;
        };

    private:
        struct CoincidenceSet* d;
};


// =============================================================================
// BlankBlockClass
// =============================================================================
class BlankBlockClass {
    public:
        // Construct BlankBlock
        BlankBlockClass(const uint32_t num_o) {
            blank_block_construct(&b, num_o);
        };

        // Deconstruct BlankBlock
        ~BlankBlockClass() {
            blank_block_destruct(&b);
        };

        // Clear block
        void clear() {
            blank_block_clear(&b);
        }

        // Get output Page object
        PageClass* get_output() {
            return new PageClass(b.output);
        };

    private:
        struct BlankBlock b;
};


// =============================================================================
// ScalarEncoderClass
// =============================================================================
class ScalarEncoderClass {
    public:
        // Construct ScalarEncoder
        ScalarEncoderClass(
                const double min_val,
                const double max_val,
                const uint32_t num_s,
                const uint32_t num_as) {
            scalar_encoder_construct(&e, min_val, max_val, num_s, num_as);
        };

        // Deconstruct ScalarEncoder
        ~ScalarEncoderClass() {
            scalar_encoder_destruct(&e);
        };

        // Clear block
        void clear() {
            scalar_encoder_clear(&e);
        }

        // Compute block
        void compute(double value) {
            scalar_encoder_compute(&e, value);
        };

        // Get output Page object
        PageClass* get_output() {
            return new PageClass(e.output);
        };

    private:
        struct ScalarEncoder e;
};


// =============================================================================
// SymbolsEncoderClass
// =============================================================================
class SymbolsEncoderClass {
    public:
        // Construct SymbolsEncoder
        SymbolsEncoderClass(
                const uint32_t max_symbols,
                const uint32_t num_s) {
            symbols_encoder_construct(&e, max_symbols, num_s);
        };

        // Deconstruct SymbolsEncoder
        ~SymbolsEncoderClass() {
            symbols_encoder_destruct(&e);
        };

        // Clear block
        void clear() {
            symbols_encoder_clear(&e);
        }

        // Compute block
        void compute(const uint32_t value) {
            if (value >= e.max_symbols || value < 0) {
                throw std::range_error("Error: symbol exceeds range for this SymbolEncoder");
            }
            symbols_encoder_compute(&e, value);
        };

        // Get symbols
        std::vector<uint32_t> get_symbols() {
            std::vector<uint32_t> symbols;
            symbols.reserve(e.num_symbols);

            for (uint32_t s = 0; s < e.num_symbols; s++) {
                symbols.push_back(e.symbols[s]);
            }

            return symbols;
        };

        // Get output Page object
        PageClass* get_output() {
            return new PageClass(e.output);
        };

    private:
        struct SymbolsEncoder e;
};


// =============================================================================
// PersistenceEncoderClass
// =============================================================================
class PersistenceEncoderClass {
    public:
        // Construct PersistenceEncoder
        PersistenceEncoderClass(
                const double min_val,
                const double max_val,
                const uint32_t num_s,
                const uint32_t num_as,
                const uint32_t max_steps) {
            persistence_encoder_construct(&e, min_val, max_val, num_s, num_as, max_steps);
        };

        // Deconstruct PersistenceEncoder
        ~PersistenceEncoderClass() {
            persistence_encoder_destruct(&e);
        };

        // Reset persistence
        void reset() {
            persistence_encoder_reset(&e);
        };

        // Clear block
        void clear() {
            persistence_encoder_clear(&e);
        }

        // Compute block
        void compute(double value) {
            persistence_encoder_compute(&e, value);
        };

        // Get output Page object
        PageClass* get_output() {
            return new PageClass(e.output);
        };

    private:
        struct PersistenceEncoder e;
};


// =============================================================================
// PatternClassifierClass
// =============================================================================
class PatternClassifierClass {
    public:
        // Construct PatternClassifier
        PatternClassifierClass(
                const std::vector<uint32_t> labels,
                const uint32_t num_l,
                const uint32_t num_s,
                const uint32_t num_as,
                const uint32_t perm_thr,
                const uint32_t perm_inc,
                const uint32_t perm_dec,
                const double pct_pool,
                const double pct_conn,
                const double pct_learn) {
            // TODO: check that num_l <= to labels.size()
            //const char **clabels_ptr = clabels.data();
            pattern_classifier_construct(
                &pc,
                labels.data(),
                num_l,
                num_s,
                num_as,
                perm_thr,
                perm_inc,
                perm_dec,
                pct_pool,
                pct_conn,
                pct_learn);
        };

        // Deconstruct PatternClassifier
        ~PatternClassifierClass() {
            pattern_classifier_destruct(&pc);
        };

        // Initialize block
        void initialize() {
            pattern_classifier_initialize(&pc);
        };

        // Save block
        void save(std::string file_str) {
            pattern_classifier_save(&pc, file_str.c_str());
        };

        // Load block
        void load(std::string file_str) {
            pattern_classifier_load(&pc, file_str.c_str());
        };

        // Clear block
        void clear() {
            pattern_classifier_clear(&pc);
        }

        // Compute block
        void compute(uint32_t in_label, const uint32_t learn_flag) {
            pattern_classifier_compute(&pc, in_label, learn_flag);
        };

        // Get label probabilities
        std::vector<double> get_probabilities() {
            std::vector<double> probs;
            probs.reserve(pc.num_l);

            pattern_classifier_update_probabilities(&pc);

            for (uint32_t l = 0; l < pc.num_l; l++) {
                probs.push_back(pc.probs[l]);
            }

            return probs;
        };

        // Get a particular CoincidenceSet object
        CoincidenceSetClass* get_coincidence_set(const uint32_t d) {
            if (d > pc.num_s) {
                throw std::range_error("Error: Index out of range in get_coincidence_set()");
            }
            
            if (pc.init_flag == 0) {
                throw std::runtime_error("Error: Could not retrieve CoincidenceSet because PatternClassifier is not initialized.");
            }

            return new CoincidenceSetClass(&pc.coincidence_sets[d]);
        };

        // Get input Page object
        PageClass* get_input() {
            return new PageClass(pc.input);
        };

        // Get output Page object
        PageClass* get_output() {
            return new PageClass(pc.output);
        };

    private:
        struct PatternClassifier pc;
};


// =============================================================================
// PatternPoolerClass
// =============================================================================
class PatternPoolerClass {
    public:
        // Construct PatternPooler
        PatternPoolerClass(
                const uint32_t num_s,
                const uint32_t num_as,
                const uint32_t perm_thr,
                const uint32_t perm_inc,
                const uint32_t perm_dec,
                const double pct_pool,
                const double pct_conn,
                const double pct_learn) {
            pattern_pooler_construct(
                &pp,
                num_s,
                num_as,
                perm_thr,
                perm_inc,
                perm_dec,
                pct_pool,
                pct_conn,
                pct_learn);
        };

        // Deconstruct PatternPooler
        ~PatternPoolerClass() { 
            pattern_pooler_destruct(&pp);
        };

        // Initialize block
        void initialize() {
            pattern_pooler_initialize(&pp);
        };

        // Save block
        void save(std::string file_str) {
            pattern_pooler_save(&pp, file_str.c_str());
        };

        // Load block
        void load(std::string file_str) {
            pattern_pooler_load(&pp, file_str.c_str());
        };

        // Clear block
        void clear() {
            pattern_pooler_clear(&pp);
        }

        // Compute block
        void compute(const uint32_t learn_flag) {
            pattern_pooler_compute(&pp, learn_flag);
        };

        // Get a particular CoincidenceSet object
        CoincidenceSetClass* get_coincidence_set(const uint32_t d) {
            if (d > pp.num_s) {
                throw std::range_error("Error: Index out of range in get_coincidence_set()");
            }

            if (pp.init_flag == 0) {
                throw std::runtime_error("Error: Could not retrieve CoincidenceSet because PatternPooler is not initialized.");
            }

            return new CoincidenceSetClass(&pp.coincidence_sets[d]);
        };

        // Get input Page object
        PageClass* get_input() {
            return new PageClass(pp.input);
        };

        // Get output Page object
        PageClass* get_output() {
            return new PageClass(pp.output);
        };

    private:
        struct PatternPooler pp;
};


// =============================================================================
// SequenceLearnerClass
// =============================================================================
class SequenceLearnerClass {
    public:
        // Construct SequenceLearner
        SequenceLearnerClass(
                const uint32_t num_spc,
                const uint32_t num_dps,
                const uint32_t num_rpd,
                const uint32_t d_thresh,
                const uint32_t perm_thr,
                const uint32_t perm_inc,
                const uint32_t perm_dec) {
            sequence_learner_construct(
                &sl,
                num_spc,
                num_dps,
                num_rpd,
                d_thresh,
                perm_thr,
                perm_inc,
                perm_dec);
        };

        // Deconstruct SequenceLearner
        ~SequenceLearnerClass() {
            sequence_learner_destruct(&sl);
        };

        // Initialize block
        void initialize() {
            sequence_learner_initialize(&sl);
        };

        // Save block
        void save(std::string file_str) {
            sequence_learner_save(&sl, file_str.c_str());
        };

        // Load block
        void load(std::string file_str) {
            sequence_learner_load(&sl, file_str.c_str());
        };

        // Clear block
        void clear() {
            sequence_learner_clear(&sl);
        }

        // Compute block
        void compute(const uint32_t learn_flag) {
            sequence_learner_compute(&sl, learn_flag);
        };

        // Get abnormality score
        double get_score() {
            return sequence_learner_get_score(&sl);
        };

        // Get number of historical statelets
        uint32_t get_historical_count() {
			return sl.count_hs;
        };

        // Get number of used coincidence sets
        uint32_t get_coincidence_set_count() {
			return sl.count_hd;
        };

        // Get historical statelets
        std::vector<uint8_t> get_historical_statelets() {
            struct BitArray* ba = sequence_learner_get_historical_statelets(&sl);
            uint32_t num_bits = ba->num_bits;
            std::vector<uint8_t> bits;
            bits.reserve(num_bits);

            for (uint32_t i = 0; i < num_bits; i++) {
                bits.push_back(bitarray_get_bit(ba, i));
            }

            return bits;
        };

        // Get number of coincidence sets per statelet
        std::vector<uint32_t> get_num_coincidence_sets_per_statelet() {
            std::vector<uint32_t> array;
            array.reserve(sl.num_s);

            for (uint32_t s = 0; s < sl.num_s; s++) {
                array.push_back(sl.s_next_d[s]);
            }

            return array;
        };

        // Get a particular hidden CoincidenceSet object
        CoincidenceSetClass* get_hidden_coincidence_set(const uint32_t d) {
            if (d > sl.num_d) {
                throw std::range_error("Error: Index out of range in get_hidden_coincidence_set()");
            }

            if (sl.init_flag == 0) {
                throw std::runtime_error("Error: Could not retrieve CoincidenceSet object because SequenceLearner is not initialized.");
            }

            return new CoincidenceSetClass(&sl.d_hidden[d]);
        };

        // Get a particular output CoincidenceSet object
        CoincidenceSetClass* get_output_coincidence_set(const uint32_t d) {
            if (d > sl.num_d) {
                throw std::range_error("Error: Index out of range in get_output_coincidence_set()");
            }

            if (sl.init_flag == 0) {
                throw std::runtime_error("Error: Could not retrieve CoincidenceSet object because SequenceLearner is not initialized.");
            }

            return new CoincidenceSetClass(&sl.d_output[d]);
        };

        // Get input Page object
        PageClass* get_input() {
            return new PageClass(sl.input);
        };

        // Get hidden Page object
        PageClass* get_hidden() {
            return new PageClass(sl.hidden);
        };

        // Get output Page object
        PageClass* get_output() {
            return new PageClass(sl.output);
        };

    private:
        struct SequenceLearner sl;
};
