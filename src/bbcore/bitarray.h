#ifndef BITARRAY_H
#define BITARRAY_H

#include <stdint.h>

#define WORD_BYTES 4
#define WORD_BITS 32
typedef uint32_t word_t;

struct ActArray {
    uint32_t num_acts;
    uint8_t dirty_flag;
    uint32_t* acts;
};

struct BitArray {
    uint32_t num_words;
    uint32_t num_bits;
    word_t* words;
    struct ActArray* actarray;
};

void bitarray_construct(struct BitArray* ba, const uint32_t num_bits);
void bitarray_destruct(struct BitArray* ba);
void bitarray_resize(struct BitArray* ba, const uint32_t num_bits);
void bitarray_random_fill(struct BitArray* ba, double percent);
void bitarray_clear(struct BitArray* ba);
void bitarray_clear_bit(struct BitArray* ba, const uint32_t idx);
void bitarray_set(struct BitArray* ba, const uint32_t beg, const uint32_t end);
void bitarray_set_bit(struct BitArray* ba, const uint32_t idx);
uint32_t bitarray_get_bit(const struct BitArray* ba, const uint32_t idx);
uint32_t bitarray_count(const struct BitArray* ba);
void bitarray_not(const struct BitArray* in, struct BitArray* out);

void bitarray_and(
    const struct BitArray* in0,
    const struct BitArray* in1,
    struct BitArray* out);

void bitarray_or(
    const struct BitArray* in0,
    const struct BitArray* in1,
    struct BitArray* out);

void bitarray_xor(
    const struct BitArray* in0,
    const struct BitArray* in1,
    struct BitArray* out);

void bitarray_copy(
    struct BitArray* dst,
    const struct BitArray* src,
    const uint32_t dst_word_offset,
    const uint32_t src_word_offset,
    const uint32_t src_word_size);

struct ActArray* bitarray_get_actarray(struct BitArray* ba);

#endif