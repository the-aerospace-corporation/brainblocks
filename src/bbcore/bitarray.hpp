#ifndef BITARRAY_HPP
#define BITARRAY_HPP

#include <cstdint>
#include <vector>

#define WORD_BYTES 4
#define WORD_BITS 32
typedef uint32_t word_t;

// TODO: Need to read through this again because it might have more ideas
// https://graphics.stanford.edu/~seander/bithacks.html

class BitArray {
    public:
        BitArray();
        BitArray(const uint32_t num_bits);
        void resize(const uint32_t num_bits);
        void clear();
        void clear_actarray();
        void clear_bits();
        void fill_bits();
        void random_fill(double percent);
        void random_shuffle();
        uint32_t count();
        BitArray operator~(); // binary not
        BitArray operator&(const BitArray& in); // binary and
        BitArray operator|(const BitArray& in); // binary or
        BitArray operator^(const BitArray& in); // binary xor
        // TODO see if it's possible to overload the [] operator to get/set particular bitarray element

        // getters and setters
        void set_bit(const uint32_t idx, const uint8_t val);
        uint8_t get_bit(const uint32_t idx);
        void set_bits(std::vector<uint8_t>& new_bits);
        void set_acts(std::vector<uint32_t>& new_acts);
        std::vector<uint8_t> get_bits();
        std::vector<uint32_t> get_acts();
        uint32_t get_num_bits() { return num_bits; };
        uint32_t get_num_words() { return (uint32_t)words.size(); };

        // printing
        void print_info();
        void print_bits();
        void print_acts();

    public: // TODO: needs to be public for bitarray_copy function
        bool acts_dirty_flag;
        uint32_t num_bits;
        std::vector<word_t> words;
        std::vector<uint32_t> acts;
};

// TODO: could probably put this in the class as a dst.copy_from(src) function
void bitarray_copy(
    BitArray& dst,
    const BitArray& src,
    const uint32_t dst_word_offset,
    const uint32_t src_word_offset,
    const uint32_t src_word_size);

#endif