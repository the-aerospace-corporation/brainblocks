#ifndef BITARRAY_HPP
#define BITARRAY_HPP

#include <cstdint>
#include <vector>

#define WORD_BYTES 4
#define WORD_BITS 32
typedef uint32_t word_t;

// TODO: convert all pointer arrays to vectors!!!
class BitArray {
    public:
        BitArray(const uint32_t num_bits);
        void resize(const uint32_t num_bits);
        void random_fill(double percent);
        void clear_bit(const uint32_t idx);
        void set_bit(const uint32_t idx);
        uint32_t get_bit(const uint32_t idx);
        void clear_bits();
        void set_bits(std::vector<uint8_t>& bits);
        void set_acts(std::vector<uint32_t>& acts);
        std::vector<uint8_t> get_bits();  // TODO: create "dirty" flag?
        std::vector<uint32_t> get_acts(); // TODO: create "dirty" flag?
        void print_bits();
        void print_acts();
        uint32_t count();
        BitArray operator~(); // binary not
        BitArray operator&(const BitArray& in); // binary and
        BitArray operator|(const BitArray& in); // binary or
        BitArray operator^(const BitArray& in); // binary xor

        uint32_t get_num_words() { return num_words; };
        uint32_t get_num_bits() { return num_bits; };

    public: // TODO: needs to be public for bitarray_copy function
        uint32_t num_words;
        uint32_t num_bits;
        std::vector<word_t> words;
};

void bitarray_copy(
    BitArray* dst,
    const BitArray* src,
    const uint32_t dst_word_offset,
    const uint32_t src_word_offset,
    const uint32_t src_word_size);

#endif