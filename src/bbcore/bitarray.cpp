#include "bitarray.hpp"
#include "utils.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
BitArray::BitArray(const uint32_t num_bits) {

    /// error check
    if (num_bits % WORD_BITS != 0) {
        printf("Error: BitArray not divisible by 32");
        exit(1);
    }

    // initialize BitArray variables
    this->num_bits = num_bits;
    num_words = (uint32_t)((num_bits + WORD_BITS - 1) / WORD_BITS);
    words.resize(num_words);
}

// =============================================================================
// Resize
// =============================================================================
void BitArray::resize(const uint32_t num_bits) {

    // error check
    if (num_bits % WORD_BITS != 0) {
        printf("Error: BitArray not divisible by 32");
        exit(1);
    }

    // update BitArray variables
    this->num_bits = num_bits;
    num_words = (uint32_t)((num_bits + WORD_BITS - 1) / WORD_BITS);
    words.resize(num_words);
    clear_bits();
}

// =============================================================================
// Random Fill
// =============================================================================
void BitArray::random_fill(double percent) {
    uint32_t* rand_indices = (uint32_t*)malloc(num_bits * sizeof(*rand_indices));

    for (uint32_t i = 0; i < num_bits; i++) {
        rand_indices[i] = i;
    }

    utils_shuffle(rand_indices, num_bits);

    clear_bits();

    for (uint32_t i = 0; i < (uint32_t)(num_bits * percent); i++) {
        set_bit(rand_indices[i]);
    }

    free(rand_indices);
}

// =============================================================================
// Clear Bit
// =============================================================================
void BitArray::clear_bit(const uint32_t idx) {
    words[idx / WORD_BITS] &= ~(1 << (idx % WORD_BITS));
}

// =============================================================================
// Set Bit
// =============================================================================
void BitArray::set_bit(const uint32_t idx) {
    words[idx / WORD_BITS] |= 1 << (idx % WORD_BITS);
}

// =============================================================================
// Get Bit
// =============================================================================
uint32_t BitArray::get_bit(const uint32_t idx) {
    uint32_t word_index = idx / WORD_BITS;
    word_t curr_word = words[word_index];

    uint32_t bit_index = idx % WORD_BITS;
    word_t bit_mask = 1 << bit_index;

    return (curr_word & bit_mask) != 0;
}

// =============================================================================
// Clear Bits
// =============================================================================
void BitArray::clear_bits() {
    for (uint32_t w = num_words; w-- > 0; ) {
        words[w] = 0x00000000;
    }
}

// =============================================================================
// Set Bits
// =============================================================================
void BitArray::set_bits(std::vector<uint8_t>& bits) {
    if (bits.size() > num_bits) {
        std::cout << "Warning in BitArray::set_bits(): input vector size > num_bits.  Skipping operation." << std::endl;
        return;
    }

    clear_bits();
    for (uint32_t i = 0; i < bits.size(); i++) {
        if (bits[i] > 0) {
            set_bit(i);
        }
    }
}

// =============================================================================
// Set Acts
// =============================================================================
void BitArray::set_acts(std::vector<uint32_t>& acts) {
    clear_bits();
    for (uint32_t i = 0; i < acts.size(); i++) {
        if (acts[i] > num_bits) {
            std::cout << "Warning in BitArray::set_acts(): input act value > num_bits.  Skipping this value." << std::endl;
            continue;
        }
        set_bit(acts[i]);
    }
}

// =============================================================================
// Get Bits
// =============================================================================
std::vector<uint8_t> BitArray::get_bits() {
    std::vector<uint8_t> bits(num_bits);
    for (uint32_t i = 0; i < num_bits; i++) {
        bits[i] = get_bit(i);
    }
    return bits;
}

// =============================================================================
// Get Acts
// =============================================================================
std::vector<uint32_t> BitArray::get_acts() {
    uint32_t num_acts = count();
    std::vector<uint32_t> acts(num_acts);
    uint32_t j = 0;
    for (uint32_t i = 0; i < num_bits; i++) {
        if (get_bit(i)) {
            acts[j] = i;
            j++;
        }
    }
    return acts;
}

// =============================================================================
// Print Bits
// =============================================================================
void BitArray::print_bits() {
    std::cout << "[";
    for (uint32_t i = 0; i < num_bits; i++) {
        std::cout << get_bit(i);
    }
    std::cout << "]" << std::endl;
}

// =============================================================================
// Print Acts
// =============================================================================
void BitArray::print_acts() {
    std::cout << "[";
    for (uint32_t i = 0; i < num_bits; i++) {
        if (get_bit(i)) {
            std::cout << i;
            if (i < num_bits - 1) {
                std::cout << ", ";
            }
        }
    }
    std::cout << "]" << std::endl;
}

// =============================================================================
// Count
// =============================================================================
// https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
uint32_t BitArray::count() {
    uint32_t count = 0;
    for (uint32_t w = num_words; w-- > 0; ) {
        uint32_t i = words[w];
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        count += ((((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24);
    }
    return count;
}

// =============================================================================
// Binary Not
// =============================================================================
BitArray BitArray::operator~() {
    BitArray out(num_bits);
    for (uint32_t w = out.num_words; w-- > 0; ) {
        out.words[w] = ~words[w];
    }
    return out;
}

// =============================================================================
// Binary And
// =============================================================================
BitArray BitArray::operator&(const BitArray& in) {
    BitArray out(num_bits);
    for (uint32_t w = out.num_words; w-- > 0; ) {
        out.words[w] = words[w] & in.words[w];
    }
    return out;
}

// =============================================================================
// Binary Or
// =============================================================================
BitArray BitArray::operator|(const BitArray& in) {
    BitArray out(num_bits);
    for (uint32_t w = out.num_words; w-- > 0; ) {
        out.words[w] = words[w] | in.words[w];
    }
    return out;
}

// =============================================================================
// Binary Xor
// =============================================================================
BitArray BitArray::operator^(const BitArray& in) {
    BitArray out(num_bits);
    for (uint32_t w = out.num_words; w-- > 0; ) {
        out.words[w] = words[w] ^ in.words[w];
    }
    return out;
}

// =============================================================================
// Copy
// =============================================================================
// TODO: figure out how to do a fast copy subset on a per bit offset and size
// instead of a per word offset and size
void bitarray_copy(
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