// =============================================================================
// bitarray.cpp
// =============================================================================

// =============================================================================
// # BitArray
//
// BitArrays represent arrays of bits and contains functions for manipulating
// and accessing bit information.
//
// TODO: add more to the description.
//
// ## Links
//
// - https://graphics.stanford.edu/~seander/bithacks.html
// - https://github.com/noporpoise/BitArray
// - https://www.chessprogramming.org/BitScan
// - https://www.chessprogramming.org/Bit-Twiddling
// =============================================================================
#include "bitarray.hpp"
#include "utils.hpp"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <cstring> // for memset and memcmp

using namespace BrainBlocks;

// =============================================================================
// # Popcount
//
// Returns the number of active bits in a single word.
//
// ## Example
//
// word: {00101100}
// count = popcount(word)
// count: 3
//
// ## Links
//
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
// =============================================================================
inline uint32_t popcount(word_t w) {

    w = w - ((w >> 1) & (word_t)~(word_t)0/3);
    w = (w & (word_t)~(word_t)0/15*3) + ((w >> 2) & (word_t)~(word_t)0/15*3);
    w = (w + (w >> 4)) & (word_t)~(word_t)0/255*15;
    return (word_t)(w * ((word_t)~(word_t)0/255)) >> (sizeof(word_t) - 1) * 8;
}

// =============================================================================
// # Constructor
//
// Constructs a BitArray.
// =============================================================================
BitArray::BitArray(const uint32_t n) {

    resize(n);
}

// =============================================================================
// # Resize
//
// Resizes the vector so it contains n bits.
//
// ## Example
//
// bitarray: {}
// bitarray.resize(32);
// bitarray: {00000000000000000000000000000000}
// =============================================================================
void BitArray::resize(const uint32_t n) {

    num_b = n;
    uint32_t num_words = (uint32_t)((num_b + WBITS - 1) / WBITS);
    num_bytes = num_words * WBYTES;
    words.resize(num_words);
    clear_all();
}

// =============================================================================
// # Erase
//
// Removes all words from the BitArray.
//
// ## Example
//
// bitarray: {00000000000000000000000000000000}
// bitarray.erase();
// bitarray: {}
// =============================================================================
void BitArray::erase() {

    words.clear();
    num_b = 0;
    num_bytes = 0;
}

// =============================================================================
// # Save
//
// Saves BitArray.
// =============================================================================
void BitArray::save(FILE* fptr) {

    std::fwrite(words.data(), sizeof(words[0]), words.size(), fptr);
}

// =============================================================================
// # Load
//
// Loads BitArray.
// =============================================================================
void BitArray::load(FILE* fptr) {

    std::fread(words.data(), sizeof(words[0]), words.size(), fptr);
}

// =============================================================================
// # Set Bit
//
// Sets a particular bit to 1.
//
// ## Example
//
// bitarray: {00000000000000000000000000000000}
// bitarray.set_bit(3);
// bitarray: {00010000000000000000000000000000}
//              ^
// =============================================================================
void BitArray::set_bit(const uint32_t b) {

    assert(b < num_b);
    words[get_wrd(b)] |= 1 << (get_idx(b));
}

// =============================================================================
// # Get Bit
//
// Returns the value (0 or 1) of a particular bit.
//
// ## Example
//
// bitarray: {00010000000000000000000000000000}
//              ^
// val = bitarray.get_bit(3);
// val: 1
// =============================================================================
uint8_t BitArray::get_bit(const uint32_t b) {

    assert(b < num_b);
    return (words[get_wrd(b)] >> (get_idx(b))) & 0x1;
}

// =============================================================================
// # Clear Bit
//
// Sets a particular bit to 0.
//
// ## Example
//
// bitarray: {00010000000000000000000000000000}
// bitarray.clear_bit(3);
// bitarray: {00000000000000000000000000000000}
//               ^
// =============================================================================
void BitArray::clear_bit(const uint32_t b) {

    assert(b < num_b);
    words[get_wrd(b)] &= ~(1 << (get_idx(b)));
}

// =============================================================================
// # Toggle Bit
//
// Flips a particular bit (i.e. 0 -> 1 or 1 -> 0).
//
// ## Example
//
// bitarray: {00010000000000000000000000000000}
// bitarray.toggle_bit(3);
// bitarray.toggle_bit(0);
// bitarray: {10000000000000000000000000000000}
//            ^  ^
// =============================================================================
void BitArray::toggle_bit(const uint32_t b) {

    assert(b < num_b);
    words[get_wrd(b)] ^= 1 << (get_idx(b));
}

// =============================================================================
// # Assign Bit
//
// Assigns a particular bit to a given value (0 or 1).
//
// ## Example
//
// bitarray: {11110000000000000000000000000000}
// bitarray.assign_bit(1, 0);
// bitarray.assign_bit(6, 1);
// bitarray: {10110010000000000000000000000000}
//             ^    ^
// =============================================================================
void BitArray::assign_bit(const uint32_t b, const uint8_t val) {

    assert(b < num_b);

    if (val > 0)
        set_bit(b);
    else
        clear_bit(b);
}

// =============================================================================
// # Set Range
//
// Sets a range of bits to 1.
//
// ## Example
//
// bitarray: {00000000000000000000000000000000}
// bitarray.set_range(3, 5);
// bitarray: {00011111000000000000000000000000}
//               ^^^^^
// =============================================================================
void BitArray::set_range(const uint32_t beg, const uint32_t len) {

    assert(beg + len <= num_b);

    for (uint32_t b = beg; b < beg + len; b++)
        set_bit(b);
}

// =============================================================================
// # Clear Range
//
// Sets a range of bits to 0.
//
// ## Example
//
// bitarray: {00011111000000000000000000000000}
// bitarray.clear_range(4, 5);
// bitarray: {00010000000000000000000000000000}
//                ^^^^^
// =============================================================================
void BitArray::clear_range(const uint32_t beg, const uint32_t len) {

    assert(beg + len <= num_b);

    for (uint32_t b = beg; b < beg + len; b++)
        clear_bit(b);
}

// =============================================================================
// # Toggle Range
//
// Flips a range of bits.  Similar to binary not operation applied to the range.
//
// ## Example
//
// bitarray: {00101101010000000000000000000000}
// bitarray.toggle_range(2, 6);
// bitarray: {00010010010000000000000000000000}
//              ^^^^^^
// =============================================================================
void BitArray::toggle_range(const uint32_t beg, const uint32_t len) {

    assert(beg + len <= num_b);

    for (uint32_t b = beg; b < beg + len; b++)
        toggle_bit(b);
}

// =============================================================================
// # Set All
//
// Sets all bits to 1.
//
// ## Example
//
// bitarray: {00101101010000100000000010000100}
// bitarray.set_all();
// bitarray: {11111111111111111111111111111111}
//            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// =============================================================================
void BitArray::set_all() {

    memset(words.data(), 0xFFFFFFFF, num_bytes);
}

// =============================================================================
// # Clear All
//
// Sets all bits to 0.
//
// ## Example
//
// bitarray: {00101101010000100000000010000100}
// bitarray.clear_all();
// bitarray: {00000000000000000000000000000000}
//            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// =============================================================================
void BitArray::clear_all() {

    memset(words.data(), 0x00000000, num_bytes);
}

// =============================================================================
// # Toggle All
//
// Flips all bits.  Similar to binary not operation.
//
// ## Example
//
// bitarray: {00101101010000100000000010000100}
// bitarray.clear_all();
// bitarray: {11010010101111011111111101111011}
//            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// =============================================================================
void BitArray::toggle_all() {

    for (uint32_t w = 0; w < words.size(); w++)
        words[w] = ~words[w];
}

// =============================================================================
// # Set Bits
//
// Assigns a collection of bits to given values (0 or 1).
//
// ## Example
//
// vals = {0,0,1,0,1,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0};
// bitarray.set_bits(vals);
// bitarray: {00101101010000100000000010000100}
//            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// =============================================================================
void BitArray::set_bits(std::vector<uint8_t>& vals) {

    assert(vals.size() <= num_b);

    clear_all();

    for (uint32_t i = 0; i < vals.size(); i++) {
        if (vals[i] > 0)
            set_bit(i);
    }
}

// =============================================================================
// # Set Acts
//
// Assign a collection of bit indices to 1.
//
// ## Example:
//
// idxs = {2, 4, 7, 9, 15};
// bitarray.set_acts(idxs);
// bitarray: {00101001010000010000000000000000}
//              ^ ^  ^ ^     ^
// =============================================================================
void BitArray::set_acts(std::vector<uint32_t>& idxs) {

    clear_all();

    for (uint32_t i = 0; i < idxs.size(); i++) {
        if (idxs[i] > num_b)
            continue;

        set_bit(idxs[i]);
    }
}

// =============================================================================
// # Get Bits
//
// Returns all bit values.
//
// ## Example
//
// bitarray: {00101101010000100000000010000100}
// vals = bitarray.get_bits();
// vals: {0,0,1,0,1,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0}
// =============================================================================
std::vector<uint8_t> BitArray::get_bits() {

    std::vector<uint8_t> vals(num_b);

    for (uint32_t b = 0; b < num_b; b++)
        vals[b] = get_bit(b);

    return vals;
}

// =============================================================================
// # Get Acts
//
// Returns the indices of all set (1) bits.
//
// ## Example
//
// bitarray: {00101101010000100000000010000100}
// idxs = bitarray.get_acts();
// vals: {2, 4, 5, 7, 9, 14, 24, 29}
// =============================================================================
std::vector<uint32_t> BitArray::get_acts() {

    uint32_t n = num_set();
    uint32_t beg = 0;
    uint32_t result = 0xFFFFFFFF;
    std::vector<uint32_t> idxs(n);

    for (uint32_t i = 0; i < n; i++) {
        find_next_set_bit(beg, &result);
    idxs[i] = result;
        beg = result + 1;
    }

    return idxs;
}

// =============================================================================
// # Number of Set
//
// Returns the number of set (1) bits.
//
// ## Example
//
// bitarray: {00101101010000100000000010000100}
//              ^ ^^ ^ ^    ^         ^    ^
// count = bitarray.num_set();
// count = 7
// =============================================================================
uint32_t BitArray::num_set() {

    uint32_t count = 0;

    for (uint32_t w = 0; w < words.size(); w++)
        count += popcount(words[w]);

    return count;
}

// =============================================================================
// # Number of Cleared
//
// Returns the number of clear (0) bits.
//
// ## Example
//
// bitarray: {00101101010000100000000010000100}
//            ^^ ^  ^ ^ ^^^^ ^^^^^^^^^ ^^^^ ^^
// count = bitarray.num_cleared();
// count = 25
// =============================================================================
uint32_t BitArray::num_cleared() {

    return num_b - num_set();
}

// =============================================================================
// # Number of Similar
//
// Returns the number of similar set (1) bits between two BitArrays.
//
// ## Example
//
// bitarray0: {00110000100100001110000000010100}
// bitarray1: {10010001100001000100000111000101}
//                ^    ^        ^           ^
// count = bitarray0.num_similar(bitarray1);
// count: 4
// =============================================================================
uint32_t BitArray::num_similar(const BitArray& ba) {

    assert(words.size() == ba.words.size());

    word_t word;
    uint32_t count = 0;

    for (uint32_t w = 0; w < words.size(); w++) {
        word = words[w] & ba.words[w];
        count += popcount(word);
    }

    return count;
}

// =============================================================================
// # Find Next Set Bit
//
// Finds the index of the next set bit from a bit offset.
// - Wraps around to offset-1
// - Returns 1 if the next set bit has been found
// - Returns 0 if no set bits found
//
// ## Example 0
//
// bitarray: {00110000100100001110000000010100}
//                 ^  ^
// uint32_t result;
// success = find_next_set_bit(5, *result);
// success: 1
// result: 8
//
// ## Example 1
//
// bitarray: {00000000000000000000000000000000}
//                 ^
// uint32_t result;
// success = find_next_set_bit(5, *result);
// success: 0
// result: undefined
// =============================================================================
bool BitArray::find_next_set_bit(const uint32_t beg, uint32_t* result) {

    return find_next_set_bit(beg, num_b - 1, result);
}

// =============================================================================
// # Find Next Set Bit
// =============================================================================
bool BitArray::find_next_set_bit(
        const uint32_t beg,
        const uint32_t len,
        uint32_t* result) {

    assert(beg < num_b);
    assert(len > 0 && len <= num_b);

    // Get final bit index (inclusive)
    uint32_t end = beg + len;

    if (end > num_b)
        end -= num_b;

    // Get number of words to compute
    uint32_t num_w = (uint32_t)((double)len / (double)WBITS) + 1;

    // Setup variables
    uint32_t bw = get_wrd(beg); // beg word
    uint32_t bi = get_idx(beg); // beg index
    uint32_t ew = get_wrd(end); // end word
    uint32_t ei = get_idx(end); // end index
    word_t bmask = bitmask(bi); // beg mask
    word_t emask = bitmask(ei); // end mask
    word_t word;                // working word

    // If only 1 word to compute
    if (num_w == 1) {

        // Handle high side of beg idx
        word = words[bw] & ~bmask;

        if (word > 0) {
            *result = (bw * WBITS) + trailing_zeros(word);
            return true;
        }

        // Handle low side of end idx
        word = words[bw] & emask;

        if (word > 0) {
            *result = (bw * WBITS) + trailing_zeros(word);
            return true;
        }
    }

    // If multiple words to compute
    else {

        // Handle high side of first word beg idx
        word = words[bw] & ~bmask;

        if (word > 0) {
            *result = (bw * WBITS) + trailing_zeros(word);
            return true;
        }

        // Handle middle words
        uint32_t mid = num_w - 1;

        if(bw == ew)
            mid = num_w;

        for (uint32_t i = 1; i < mid; i++) {

            // Get word index (w) by wrapping around words array if necessary
            uint32_t j = bw + i;
            uint32_t w = (j < num_w) ? j : (j - num_w);

            word = words[w];

            if (word > 0) {
                *result = (w * WBITS) + trailing_zeros(word);
                return true;
            }
        }

        // Handle low side of last word end idx
        word = words[ew] & emask;

        if (word > 0) {
            *result = (ew * WBITS) + trailing_zeros(word);
            return true;
        }
    }

    return false;
}

// =============================================================================
// # Random Shuffle
//
// Randomly shuffles the BitArray.
//
// ## Example
//
// bitarray: {01101010000000010000001010000100}
// bitarray.random_shuffle();
// bitarray: {10000001001000110000000110001000}
// =============================================================================
void BitArray::random_shuffle(std::mt19937& rng) {

    for (uint32_t i = num_b - 1; i >= 1; i--) {
        uint32_t j = rng() % (i + 1);
        uint32_t temp = get_bit(i);
        assign_bit(i, get_bit(j));
        assign_bit(j, temp);
    }
}

// =============================================================================
// # Random Set from Number
//
// Randomly set a particular collection of bits to 1 by a given number.
//
// ## Example
//
// bitarray.random_set_num(8);
// bitarray: {01101010000000010000001010000100}
// =============================================================================
void BitArray::random_set_num(std::mt19937& rng, const uint32_t num) {

    assert(num <= num_b);

    clear_all();

    for (uint32_t i = 0; i < num; i++)
        set_bit(i);

    random_shuffle(rng);
}

// =============================================================================
// # Random Set from Percentage
//
// Randomly set a particular collection of bits to 1 by percentage (0% to 100%)
// of the total number of bits.
//
// ## Example
//
// bitarray.random_set_pct(0.25);
// bitarray: {01101010000000010000001010000100}
// =============================================================================
void BitArray::random_set_pct(std::mt19937& rng, const double pct) {

    assert(pct >= 0.0 && pct <= 1.0);

    clear_all();

    for (uint32_t i = 0; i < (uint32_t)(num_b * pct); i++)
        set_bit(i);

    random_shuffle(rng);
}

// =============================================================================
// # Binary Not Operator
//
// Returns a BitArray after performing a "binary not" on this BitArray.
//
// ## Example:
//
// bitarray0: {01101010000000010000001010000100}
// bitarray1 = ~bitarray0;
// bitarray1: {10010101111111101111110101111011}
// =============================================================================
BitArray BitArray::operator~() {

    BitArray out(num_b);

    for (uint32_t w = 0; w < words.size(); w++)
        out.words[w] = ~words[w];

    return out;
}

// =============================================================================
// # Binary And Operator
//
// Returns a BitArray after performing a "binary and" on this BitArray and an
// inputted BitArray.
//
// ## Example:
//
// bitarray0: {01101010110010010000111110000100}
// bitarray1: {00100010010100010000001010100000}
//               ^   ^  ^     ^      ^ ^
// bitarray2 = bitarray0 & bitarray1;
// bitarray2: {00100010010000010000001010000000}
// =============================================================================
BitArray BitArray::operator&(const BitArray& in) {

    BitArray out(num_b);

    for (uint32_t w = 0; w < words.size(); w++)
        out.words[w] = words[w] & in.words[w];

    return out;
}

// =============================================================================
// # Binary Or Operator
//
// Returns a BitArray after performing a "binary or" on this BitArray and an
// inputted BitArray.
//
// ## Example:
//
// bitarray0: {01101010110010010000111110000100}
// bitarray1: {00100010010100010000001010100000}
//             ^^ ^ ^ ^^ ^^  ^    ^^^^^ ^  ^
// bitarray2 = bitarray0 | bitarray1;
// bitarray2: {01101010110110010000111110100100}
// =============================================================================
BitArray BitArray::operator|(const BitArray& in) {

    BitArray out(num_b);

    for (uint32_t w = 0; w < words.size(); w++)
        out.words[w] = words[w] | in.words[w];

    return out;
}

// =============================================================================
// # Binary Xor Operator
//
// Returns a BitArray after performing a "binary xor" on this BitArray and an
// inputted BitArray.
//
// ## Example:
//
// bitarray0: {01101010110010010000111110000100}
// bitarray1: {00100010010100010000001010100000}
//              ^  ^   ^  ^^       ^^ ^  ^  ^
// bitarray2 = bitarray0 ^ bitarray1;
// bitarray2: {01001000100110000000110100100100}
// =============================================================================
BitArray BitArray::operator^(const BitArray& in) {

    BitArray out(num_b);

    for (uint32_t w = 0; w < words.size(); w++)
        out.words[w] = words[w] ^ in.words[w];

    return out;
}

// =============================================================================
// # Is Equal Operator
//
// Returns true if two BitArrays are equal, otherwise returns false.
//
// ## Example
//
// bitarray0: {11111111000000000000000000000000}
// bitarray1: {11111111000000000000000000000000}
// bool is_equal = bitarray0 == bitarray1;
// is_equal: true
// =============================================================================
bool BitArray::operator==(const BitArray& in) {

    int n = memcmp(words.data(), in.words.data(), words.size() * WBYTES);

    if (n == 0)
        return true;
    else
        return false;
}

// =============================================================================
// # Is Not Equal Operator
//
// Returns true if two BitArrays are not equal, otherwise returns false.
//
// ## Example
//
// bitarray0: {11111111000000000000000000000000}
// bitarray1: {00000000000000000000000011111111}
// bool is_not_equal = bitarray0 != bitarray1;
// is_not_equal: true
// =============================================================================
bool BitArray::operator!=(const BitArray& in) {

    int n = memcmp(words.data(), in.words.data(), words.size() * WBYTES);

    if (n != 0)
        return true;
    else
        return false;
}

// =============================================================================
// # Print Bits
//
// Prints all bit values to the terminal.
//
// ## Example
//
// bitarray: {11111111000000000000000000000000}
// bitarray.print_bits();
// {11111111000000000000000000000000}
// =============================================================================
void BitArray::print_bits() {

    std::cout << "{";

    for (uint32_t i = 0; i < num_b; i++)
        std::cout << (uint32_t)get_bit(i);

    std::cout << "}" << std::endl;
}

// =============================================================================
// # Print Acts
//
// Prints all set bit (1) indices to the terminal.
//
// ## Example
//
// bitarray: {11111111000000000000000000000000}
// bitarray.print_acts();
// {0, 1, 2, 3, 4, 5, 6, 7}
// =============================================================================
void BitArray::print_acts() {

    std::vector<uint32_t> acts = get_acts();
    uint32_t num_acts = (uint32_t)acts.size();

    std::cout << "{";

    for (uint32_t i = 0; i < num_acts; i++) {
        std::cout << acts[i];

        if (i < num_acts - 1)
            std::cout << ", ";
    }

    std::cout << "}" << std::endl;
}

// =============================================================================
// # Memory Usage (in bytes)
//
// Returns an estimate of the number of bytes used by this BitArray.
//
// ## Example
//
// nbytes = bitarray.memory_usage()
// nbytes: 132
// =============================================================================
uint32_t BitArray::memory_usage() {

    uint32_t bytes = 0;

    bytes += sizeof(num_b);
    bytes += sizeof(num_bytes);
    bytes += num_bytes;

    return bytes;
}
