#include "bitarray.hpp"
#include "utils.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
BitArray::BitArray() {
    num_bits = 0;
    acts_dirty_flag = true;
}

// =============================================================================
// Constructor
// =============================================================================
BitArray::BitArray(const uint32_t num_bits) {
    //if (num_bits % WORD_BITS != 0) {
    //    printf("Warning: BitArray not divisible by 32");
    //}

    this->num_bits = num_bits;
    uint32_t num_words = (uint32_t)((num_bits + WORD_BITS - 1) / WORD_BITS);
    words.resize(num_words);
    acts_dirty_flag = true;
}

// =============================================================================
// Resize
// =============================================================================
void BitArray::resize(const uint32_t num_bits) {
    //if (num_bits % WORD_BITS != 0) {
    //    printf("Warning: BitArray not divisible by 32");
    //}

    this->num_bits = num_bits;
    uint32_t num_words = (uint32_t)((num_bits + WORD_BITS - 1) / WORD_BITS);
    words.resize(num_words);
    clear_bits();
    acts_dirty_flag = true;
}

// =============================================================================
// Clear
// =============================================================================
void BitArray::clear() {
    words.clear();
    acts.clear();
    num_bits = 0;
    acts_dirty_flag = true;
}

// =============================================================================
// Clear ActArray
// =============================================================================
void BitArray::clear_actarray() {
    acts.clear();
    acts_dirty_flag = true;
}


// =============================================================================
// Clear Bits
// =============================================================================
void BitArray::clear_bits() {
    for (uint32_t w = (uint32_t)words.size(); w-- > 0; ) {
        words[w] = 0x00000000;
    }

    acts_dirty_flag = true;
}

// =============================================================================
// Fill Bits
// =============================================================================
void BitArray::fill_bits() {
    for (uint32_t w = (uint32_t)words.size(); w-- > 0; ) {
        words[w] = 0xFFFFFFFF;
    }

    acts_dirty_flag = true;
}

// =============================================================================
// Random Fill
// =============================================================================
void BitArray::random_fill(double percent) {
    clear_bits();
    
    for (uint32_t i = 0; i < (uint32_t)(num_bits * percent); i++) {
        set_bit(i, 1);
    }

    random_shuffle();

    acts_dirty_flag = true;
}

// =============================================================================
// Random Shuffle
// =============================================================================
void BitArray::random_shuffle() {
    for (uint32_t i = num_bits - 1; i >= 1; i--) {
        uint32_t j = rand() % (i + 1);
        uint32_t temp = get_bit(i);
        set_bit(i, get_bit(j));
        set_bit(j, temp);
    }

    acts_dirty_flag = true;
}

// =============================================================================
// Count
// =============================================================================
// https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
uint32_t BitArray::count() {
    uint32_t count = 0;
    for (uint32_t w = (uint32_t)words.size(); w-- > 0; ) {
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
    for (uint32_t w = (uint32_t)out.words.size(); w-- > 0; ) {
        out.words[w] = ~words[w];
    }
    
    out.acts_dirty_flag = true;
    return out;
}

// =============================================================================
// Binary And
// =============================================================================
BitArray BitArray::operator&(const BitArray& in) {
    BitArray out(num_bits);
    for (uint32_t w = (uint32_t)out.words.size(); w-- > 0; ) {
        out.words[w] = words[w] & in.words[w];
    }

    out.acts_dirty_flag = true;
    return out;
}

// =============================================================================
// Binary Or
// =============================================================================
BitArray BitArray::operator|(const BitArray& in) {
    BitArray out(num_bits);
    for (uint32_t w = (uint32_t)out.words.size(); w-- > 0; ) {
        out.words[w] = words[w] | in.words[w];
    }

    out.acts_dirty_flag = true;
    return out;
}

// =============================================================================
// Binary Xor
// =============================================================================
BitArray BitArray::operator^(const BitArray& in) {
    BitArray out(num_bits);
    for (uint32_t w = (uint32_t)out.words.size(); w-- > 0; ) {
        out.words[w] = words[w] ^ in.words[w];
    }

    out.acts_dirty_flag = true;
    return out;
}


// =============================================================================
// Set Bit
// =============================================================================
void BitArray::set_bit(const uint32_t idx, const uint8_t val) {
    if (val > 0) {
        words[idx / WORD_BITS] |= 1 << (idx % WORD_BITS);
    }
    else {
        words[idx / WORD_BITS] &= ~(1 << (idx % WORD_BITS));
    }

    acts_dirty_flag = true;
}

// =============================================================================
// Get Bit
// =============================================================================
uint8_t BitArray::get_bit(const uint32_t idx) {
    uint32_t word_index = idx / WORD_BITS;
    word_t curr_word = words[word_index];

    uint32_t bit_index = idx % WORD_BITS;
    word_t bit_mask = 1 << bit_index;

    return (curr_word & bit_mask) != 0;
}

// =============================================================================
// Set Bits
// =============================================================================
void BitArray::set_bits(std::vector<uint8_t>& new_bits) {
    if (new_bits.size() > num_bits) {
        std::cout << "Warning in BitArray::set_bits(): input vector size > num_bits.  Skipping operation." << std::endl;
        return;
    }

    clear_bits();
    for (uint32_t i = 0; i < new_bits.size(); i++) {
        if (new_bits[i] > 0) {
            set_bit(i, 1);
        }
    }

    acts_dirty_flag = true;
}

// =============================================================================
// Set Acts
// =============================================================================
void BitArray::set_acts(std::vector<uint32_t>& new_acts) {
    uint32_t num_acts = (uint32_t)new_acts.size();
    acts.resize(num_acts);
    clear_bits();

    for (uint32_t i = 0; i < num_acts; i++) {
        if (new_acts[i] > num_bits) {
            std::cout << "Warning in BitArray::set_acts(): new_act[i] > num_bits.  Skipping this activation." << std::endl;
            continue;
        }

        acts[i] = new_acts[i];
        set_bit(new_acts[i], 1);
    }

    acts_dirty_flag = false;
}

// =============================================================================
// Get Bits
// =============================================================================
std::vector<uint8_t> BitArray::get_bits() {
    std::vector<uint8_t> out_bits(num_bits);
    for (uint32_t i = 0; i < num_bits; i++) {
        out_bits[i] = get_bit(i);
    }

    return out_bits;
}

// =============================================================================
// Get Acts
// =============================================================================
std::vector<uint32_t> BitArray::get_acts() {
    if (acts_dirty_flag) {
        uint32_t num_acts = count();
        acts.resize(num_acts);
        
        uint32_t j = 0;
        for (uint32_t i = 0; i < num_bits; i++) {
            if (get_bit(i)) {
                acts[j] = i;
                j++;
            }
        }

        acts_dirty_flag = false;
    }

    return acts;
}

// =============================================================================
// Print Information
// =============================================================================
void BitArray::print_info() {
    std::cout << "{"<< std::endl;
    std::cout << "    \"object\": BitArray," << std::endl;
    std::cout << "    \"address\": 0x" << this << "," << std::endl;
    std::cout << "    \"num_bits\": " << num_bits << "," << std::endl;
    std::cout << "    \"num_words\": " << words.size() << "," << std::endl;
    std::cout << "}," << std::endl;
}

// =============================================================================
// Print Bits
// =============================================================================
void BitArray::print_bits() {
    std::cout << "{";
    for (uint32_t i = 0; i < num_bits; i++) {
        std::cout << (uint32_t)get_bit(i);
    }
    std::cout << "}" << std::endl;
}

// =============================================================================
// Print Acts
// =============================================================================
void BitArray::print_acts() {
    get_acts();

    std::cout << "{";
    uint32_t num_acts = (uint32_t)acts.size();
    for (uint32_t i = 0; i < num_acts; i++) {
        std::cout << acts[i];
        if (i < num_acts - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "}" << std::endl;
}

// =============================================================================
// Copy
// =============================================================================
// TODO: figure out how to do a fast copy subset on a per bit offset and size
// instead of a per word offset and size
void bitarray_copy(
    BitArray& dst,
    const BitArray& src,
    const uint32_t dst_word_offset,
    const uint32_t src_word_offset,
    const uint32_t src_word_size) {

    uint32_t s = src_word_offset;
    uint32_t dst_end = dst_word_offset + src_word_size;

    for (uint32_t d = dst_word_offset; d < dst_end; d++) {
        dst.words[d] = src.words[s];
        s++;
    }
    
    dst.acts_dirty_flag = true;
}