#ifndef PAGE_H
#define PAGE_H

#include "bitarray.h"
#include <stdint.h>

#define CURR 0
#define PREV 1

struct Page {
    uint32_t num_children;       // number of children
    uint32_t num_history;        // length of history
    uint32_t num_bits;           // number of bits per bitarray
    uint32_t curr;               // current bitarrays index
    uint32_t prev;               // previous bitarrays index
    uint8_t changed_flag;        // changed flag
    uint8_t init_flag;           // initialized flag
    struct Page** children;      // child page object array
    struct BitArray** bitarrays; // bitarrays 2d array
    struct BitArray* delta_ba;   // helper bitarray for computing change
};

void page_construct(
    struct Page* p,
    const uint32_t num_history,
    const uint32_t num_bits);

void page_destruct(struct Page* p);
void page_initialize(struct Page* p);
void page_add_child(struct Page* p, struct Page* child);
void page_step(struct Page* p);
void page_fetch(struct Page* p);
void page_compute_changed(struct Page* p);
void page_copy_previous_to_current(struct Page* p);
void page_clear_bits(struct Page* p, const uint32_t t);
void page_set_bit(struct Page* p, const uint32_t t, const uint32_t bit);
uint32_t page_get_bit(struct Page* p, const uint32_t t, const uint32_t bit);
struct BitArray* page_get_bitarray(const struct Page* p, const uint32_t t);
struct ActArray* page_get_actarray(const struct Page* p, const uint32_t t);
int page_idx_(const struct Page* p, const int t);

#endif