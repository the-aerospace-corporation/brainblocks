#ifndef PAGE_HPP
#define PAGE_HPP

#include "bitarray.hpp"
#include <cstdint>
#include <vector>

class Page {
    public:
        Page(const uint32_t num_history, const uint32_t num_bits);
        ~Page();
        void initialize();
        void add_child(Page& child);
        Page* get_child(const uint32_t child_idx);
        void clear_bit(const uint32_t t, const uint32_t idx);
        void set_bit(const uint32_t t, const uint32_t idx);
        uint32_t get_bit(const uint32_t t, const uint32_t idx);
        void clear_bits(const uint32_t t);
        void set_bits(const uint32_t t, std::vector<uint8_t>& bits);
        void set_acts(const uint32_t t, std::vector<uint32_t>& acts);
        std::vector<uint8_t> get_bits(const uint32_t t);
        std::vector<uint32_t> get_acts(const uint32_t t);
        void print_bits(const uint32_t t);
        void print_acts(const uint32_t t);
        void step();
        void fetch();
        void compute_changed();
        void copy_previous_to_current();

        bool has_changed() { return changed_flag; };
        bool is_initialized() { return init_flag; };
        uint32_t get_num_bits() { return num_bits; };
        uint32_t get_num_children() { return (uint32_t)children.size(); };
        uint32_t get_num_history() { return (uint32_t)bitarrays.size(); }; // TODO: rename?

    private:
        int get_index(const int t);

        uint32_t num_bits; // number of bits per bitarray
        uint32_t curr; // current bitarrays index
        uint32_t prev; // previous bitarrays index
        bool changed_flag; // changed flag
        bool init_flag; // initialized flag
        std::vector<Page*> children; // child page object array
        std::vector<BitArray*> bitarrays; // bitarray history // TODO: make smart pointer
};

#endif