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

        // setters and getters
        void set_changed_flag(const bool flag) { changed_flag = flag; };
        void set_num_bits(const uint32_t num_bits) { this->num_bits = num_bits; };
        void set_num_bitarrays(const uint32_t num_bitarrays) { this->num_bitarrays = num_bitarrays; };
        uint32_t get_num_bits() { return num_bits; };        
        uint32_t get_num_bitarrays() { return (uint32_t)bitarrays.size(); };
        uint32_t get_num_children() { return (uint32_t)children.size(); };
        bool has_changed() { return changed_flag; };
        bool is_initialized() { return init_flag; };
        //Page* get_child(const uint32_t child_idx);
        BitArray& operator[](const uint32_t t);

        // printing
        void print_info();

    private:
        int get_index(const int t);

    private:
        uint32_t num_bits; // number of bits per bitarray // TODO: needed? can get this from bitarrays
        uint32_t num_bitarrays; // number of time steps of bitarray information // TODO: not necessary anymore
        uint32_t curr; // current bitarrays index
        uint32_t prev; // previous bitarrays index
        bool changed_flag; // changed flag
        bool init_flag; // initialized flag
        std::vector<Page*> children; // child page object array //TODO: make smart pointer
        std::vector<BitArray> bitarrays; // a vector containing a time series of bitarrays
};

#endif