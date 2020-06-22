#include "bitarray.h"

#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
void bitarray_construct(struct BitArray* ba, const uint32_t num_bits) {

    /// error check
    if (num_bits % WORD_BITS != 0) {
        printf("Error: BitArray not divisible by 32");
        exit(1);
    }

    // initialize BitArray variables
    ba->num_bits = num_bits;
    ba->num_words = (uint32_t)((ba->num_bits + WORD_BITS - 1) / WORD_BITS);
    ba->words = calloc(ba->num_words, sizeof(*ba->words));
    ba->actarray = malloc(sizeof(*ba->actarray));
    ba->actarray->acts = NULL;
    ba->actarray->num_acts = 0;
    ba->actarray->dirty_flag = 1;
}

// =============================================================================
// Destructor
// =============================================================================
void bitarray_destruct(struct BitArray* ba) {
    if (ba->actarray->acts != NULL) {
        free(ba->actarray->acts);
    }

    free(ba->words);
    free(ba->actarray);
}

// =============================================================================
// Resize
// =============================================================================
void bitarray_resize(struct BitArray* ba, const uint32_t num_bits) {

    // error check
    if (num_bits % WORD_BITS != 0) {
        printf("Error: BitArray not divisible by 32");
        exit(1);
    }

    // update BitArray variables
    ba->num_bits = num_bits;
    ba->num_words = (uint32_t)((ba->num_bits + WORD_BITS - 1) / WORD_BITS);
    ba->words = realloc(ba->words, ba->num_words * sizeof(*ba->words));
    bitarray_clear(ba);
    ba->actarray = malloc(sizeof(*ba->actarray));
    ba->actarray->acts = NULL;
    ba->actarray->num_acts = 0;
    ba->actarray->dirty_flag = 1;
}

// =============================================================================
// Random Fill
// =============================================================================
void bitarray_random_fill(struct BitArray* ba, double percent) {
    ba->actarray->dirty_flag = 1;

    uint32_t* rand_indices = malloc(ba->num_bits * sizeof(*rand_indices));

    for (uint32_t i = 0; i < ba->num_bits; i++) {
        rand_indices[i] = i;
    }

    utils_shuffle(rand_indices, ba->num_bits);

    bitarray_clear(ba);

    for (uint32_t i = 0; i < (uint32_t)(ba->num_bits * percent); i++) {
        bitarray_set_bit(ba, rand_indices[i]);
    }

    free(rand_indices);

    ba->actarray->dirty_flag = 1;
}

// =============================================================================
// Clear
// =============================================================================
void bitarray_clear(struct BitArray* ba) {
    ba->actarray->dirty_flag = 1;
    for (uint32_t w = ba->num_words; w-- > 0; ) {
        ba->words[w] = 0x00000000;
    }
}

// =============================================================================
// Clear Bit
// =============================================================================
void bitarray_clear_bit(struct BitArray* ba, const uint32_t idx) {
    ba->words[idx / WORD_BITS] &= ~(1 << (idx % WORD_BITS));
    ba->actarray->dirty_flag = 1;
}

// =============================================================================
// Set
// =============================================================================
void bitarray_set(struct BitArray* ba, const uint32_t beg, const uint32_t end) {
    for (uint32_t i = beg; i <= end; i++) {
        bitarray_set_bit(ba, i);
    }

    ba->actarray->dirty_flag = 1;
}

// =============================================================================
// Set Bit
// =============================================================================
void bitarray_set_bit(struct BitArray* ba, const uint32_t idx) {
    ba->words[idx / WORD_BITS] |= 1 << (idx % WORD_BITS);
    ba->actarray->dirty_flag = 1;
}

// =============================================================================
// Get Bit
// =============================================================================
uint32_t bitarray_get_bit(const struct BitArray* ba, const uint32_t idx) {
    uint32_t word_index = idx / WORD_BITS;
    word_t curr_word = ba->words[word_index];

    uint32_t bit_index = idx % WORD_BITS;
    word_t bit_mask = 1 << bit_index;

    return (curr_word & bit_mask) != 0;
}

// =============================================================================
// Count
// =============================================================================
// https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
uint32_t bitarray_count(const struct BitArray* ba) {
    uint32_t count = 0;
    for (uint32_t w = ba->num_words; w-- > 0; ) {
        uint32_t i = ba->words[w];
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        count += ((((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24);
    }
    return count;
}

// =============================================================================
// Not
// =============================================================================
void bitarray_not(const struct BitArray* in, struct BitArray* out) {
    out->actarray->dirty_flag = 1;
    for (uint32_t w = out->num_words; w-- > 0; ) {
        out->words[w] = ~in->words[w];
    }

    out->actarray->dirty_flag = 1;
}

// =============================================================================
// And
// =============================================================================
void bitarray_and(
        const struct BitArray* in0,
        const struct BitArray* in1,
        struct BitArray* out) {

    for (uint32_t w = out->num_words; w-- > 0; ) {
        out->words[w] = in0->words[w] & in1->words[w];
    }

    out->actarray->dirty_flag = 1;
}

// =============================================================================
// Or
// =============================================================================
void bitarray_or(
        const struct BitArray* in0,
        const struct BitArray* in1,
        struct BitArray* out) {

    for (uint32_t w = out->num_words; w-- > 0; ) {
        out->words[w] = in0->words[w] | in1->words[w];
    }

    out->actarray->dirty_flag = 1;
}

// =============================================================================
// Xor
// =============================================================================
void bitarray_xor(
        const struct BitArray* in0,
        const struct BitArray* in1,
        struct BitArray* out) {

    for (uint32_t w = out->num_words; w-- > 0; ) {
        out->words[w] = in0->words[w] ^ in1->words[w];
    }

    out->actarray->dirty_flag = 1;
}

// =============================================================================
// Copy
// =============================================================================
// TODO: figure out how to do a fast copy subset on a per bit offset and size
// instead of a per word offset and size
void bitarray_copy(
        struct BitArray* dst,
        const struct BitArray* src,
        const uint32_t dst_word_offset,
        const uint32_t src_word_offset,
        const uint32_t src_word_size) {

    uint32_t s = src_word_offset;
    uint32_t dst_end = dst_word_offset + src_word_size;

    for (uint32_t d = dst_word_offset; d < dst_end; d++) {
        dst->words[d] = src->words[s];
        s++;
    }

    dst->actarray->dirty_flag = 1;
}

// =============================================================================
// Get ActArray
// =============================================================================
struct ActArray* bitarray_get_actarray(struct BitArray* ba) {
    struct ActArray* aa = ba->actarray;

    if (aa->dirty_flag) {
        aa->num_acts = bitarray_count(ba);
        aa->acts = realloc(aa->acts, aa->num_acts * sizeof(*aa->acts));

        uint32_t j = 0;
        for (uint32_t i = 0; i < ba->num_bits; i++) {
            if (bitarray_get_bit(ba, i)) {
                aa->acts[j] = i;
                j++;
            }
        }

        aa->dirty_flag = 0;
    }

    return aa;
}