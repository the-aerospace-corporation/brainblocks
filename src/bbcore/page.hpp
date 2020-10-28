#ifndef PAGE_HPP
#define PAGE_HPP

#define CURR 0
#define PREV 1

#include "bitarray.hpp"
#include <cstdint>
#include <vector>

class Page {
    public:
        Page();
        void initialize();
        void add_child(Page& child);
        void step();
        void fetch();
        void compute_changed();
        void copy_previous_to_current();
        Page* get_child(const uint32_t child_idx);
        BitArray& operator[](const uint32_t t);

        void set_num_bits(const uint32_t num_bits) { this->num_bits = num_bits; };
        void set_num_history(const uint32_t num_history) { this->num_history = num_history; };
        void set_changed_flag(const bool flag) { changed_flag = flag; };
        uint32_t get_num_bits() { return num_bits; };        
        uint32_t get_num_history() { return (uint32_t)bitarrays.size(); }; // TODO: rename?
        uint32_t get_num_children() { return (uint32_t)children.size(); };
        bool has_changed() { return changed_flag; };
        bool is_initialized() { return init_flag; };

    private:
        int get_index(const int t);

        uint32_t num_bits; // number of bits per bitarray
        uint32_t num_history; // number of time steps of bitarray information
        uint32_t curr; // current bitarrays index
        uint32_t prev; // previous bitarrays index
        bool changed_flag; // changed flag
        bool init_flag; // initialized flag
        std::vector<Page*> children; // child page object array
        std::vector<BitArray> bitarrays; // bitarray history
};

#endif