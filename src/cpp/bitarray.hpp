// =============================================================================
// bitarray.hpp
// =============================================================================
#ifndef BITARRAY_HPP
#define BITARRAY_HPP

#include <vector>
#include <cstdint>
#include <random>

namespace BrainBlocks {

// Setup 32-bit words
typedef uint32_t word_t;
#define WMAX (UINT32_MAX)
#define get_wrd(pos) ((pos) >> 5)
#define get_idx(pos) ((pos) & 31)

// TODO: get 64-bit words to work!!!
/*
// Setup 64-bit words
typedef uint64_t word_t;
#define WMAX (UINT64_MAX)
#define get_wrd(pos) ((pos) >> 6)
#define get_idx(pos) ((pos) & 63)
*/

#define WBYTES (sizeof(word_t))
#define WBITS (8 * WBYTES)
#define bitmask(nbits) ((nbits) ? ~(word_t)0 >> (WBITS-(nbits)): (word_t)0)

class BitArray {

public:

    // Constructors, destructor, resize, erase, save, load
    BitArray() {};
    BitArray(const uint32_t n);
    ~BitArray() {};
    void resize(const uint32_t n);
    void erase();
    void save(FILE* fptr);
    void load(FILE* fptr);

    // Access and manipulate a single bit
    void set_bit(const uint32_t b);
    uint8_t get_bit(const uint32_t b);
    void clear_bit(const uint32_t b);
    void toggle_bit(const uint32_t b);
    void assign_bit(const uint32_t b, const uint8_t val);

    // Manipulate all bits in a range
    void set_range(const uint32_t beg, const uint32_t len);
    void clear_range(const uint32_t beg, const uint32_t len);
    void toggle_range(const uint32_t beg, const uint32_t len);

    // Access and manipulate all bits
    void set_all();
    void clear_all();
    void toggle_all();

    // Access and manipulate bits from vectors
    void set_bits(std::vector<uint8_t>& vals);
    void set_acts(std::vector<uint32_t>& idxs);
    std::vector<uint8_t> get_bits();
    std::vector<uint32_t> get_acts();

    // Get count of bits
    uint32_t num_set();
    uint32_t num_cleared();
    uint32_t num_similar(const BitArray& ba);
    // TODO: uint32_t num_different(const BitArray& ba);
    // TODO: uint32_t hamming_distance();

    // Find indices of set/clear bits
    bool find_next_set_bit(const uint32_t beg, uint32_t* result);

    bool find_next_set_bit(
        const uint32_t beg,
        const uint32_t len,
        uint32_t* result);

    // TODO: bool find_prev_set_bit(uint32_t offset, uint32_t* result);
    // TODO: bool find_next_clear_bit(uint32_t offset, uint32_t* result);
    // TODO: bool find_prev_clear_bit(uint32_t offset, uint32_t* result);


    // Shifts and cycles
    // TODO: void shift_right();
    // TODO: void shift_left();
    // TODO: void cycle_right();
    // TODO: void cycle_left();

    // Random
    void random_shuffle(std::mt19937& rng);
    void random_set_num(std::mt19937& rng, const uint32_t num);
    void random_set_pct(std::mt19937& rng, const double percent);

    // Logic operators
    BitArray operator~();                   // binary not
    BitArray operator&(const BitArray& in); // binary and
    BitArray operator|(const BitArray& in); // binary or
    BitArray operator^(const BitArray& in); // binary xor

    // Comparisons
    bool operator==(const BitArray& in); // equals
    bool operator!=(const BitArray& in); // not equals

    // Printers
    void print_bits();
    void print_acts();

    // Get Information
    uint32_t num_bits() { return num_b; };
    uint32_t num_words() { return (uint32_t)words.size(); };
    uint32_t memory_usage();

public: // TODO: make private after figuring out bitarray_copy

    uint32_t num_b = 0;
    uint32_t num_bytes = 0;
    std::vector<word_t> words;
};

// =============================================================================
// # BitArray Copy
//
// TODO: could probably put this in the class as a dst.copy_from(src) function
// TODO: needs bit-indexing instead of word-indexing
// TODO: look into memcpy instead of using loops
// =============================================================================
inline void bitarray_copy(
        BitArray* dst,
        const BitArray* src,
        const uint32_t dst_word_offset,
        const uint32_t src_word_offset,
        const uint32_t src_word_size) {

    uint32_t s = src_word_offset;
    uint32_t dst_end = dst_word_offset + src_word_size;

    for (uint32_t d = dst_word_offset; d < dst_end; d++) {
        dst->words[d] = src->words[s];
        s++;
    }
}

// =============================================================================
// # Trailing Zeros
//
// - https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
// =============================================================================
#if defined(_WIN32)
inline int trailing_zeros(word_t word) {
    unsigned long ret;
    _BitScanForward64(&ret, word);
    return (int)ret;
}
#else
inline int trailing_zeros(word_t word) {
    return __builtin_ctzll(word);
}
#endif

// =============================================================================
// # Leading Zeros
//
// - https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
// =============================================================================
#if defined(_WIN32)
inline int leading_zeros(word_t word) {
    unsigned long ret;
    _BitScanReverse64(&ret, word);
    return (int)ret;
}
#else
inline int leading_zeros(word_t word) {
    return __builtin_clzll(word);
}
#endif

} // namespace BrainBlocks

#endif // BITARRAY_HPP
