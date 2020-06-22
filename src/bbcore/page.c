#include "page.h"

#include "bitarray.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
void page_construct(
        struct Page* p,
        const uint32_t num_history,
        const uint32_t num_bits) {

    // initialize variables
    p->num_children = 0;
    p->num_history = num_history;
    p->num_bits = num_bits;
    p->curr = 0;
    p->prev = 1;
    p->changed_flag = 1;
    p->init_flag = 0;
    p->children = NULL;
    p->bitarrays = NULL;
    p->delta_ba = NULL;
}

// =============================================================================
// Destructor
// =============================================================================
void page_destruct(struct Page* p) {

    // cleanup initialized pointers if applicable
    if (p->init_flag == 1) {

        // destruct and free each element in bitarrays
        for(uint32_t i = 0; i < p->num_history; i++) {
            bitarray_destruct(p->bitarrays[i]);
            free(p->bitarrays[i]);
        }

        // destruct helper bitarray
        bitarray_destruct(p->delta_ba);
    }

    // free pointers
    free(p->children);
    free(p->bitarrays);
    free(p->delta_ba);
}

// =============================================================================
// Initialize
// =============================================================================
void page_initialize(struct Page* p) {

    // update num_bits from children
    for (uint32_t c = 0; c < p->num_children; c++) {
        struct Page* child = p->children[c];
        
        if (child->init_flag == 0) {
            page_initialize(child);
        }
        
        p->num_bits += child->num_bits;
    }

    // TODO: might need better way of handling this.  User probably forgot
    // to attach blocks to this block and it's left hanging and alone
    // Page num_bits must be greater than 0
    if (p->num_bits == 0) {
        perror("Error: Page num_bits must be greater than 0.");
        exit(1);
    }

    // BitArrays num_bits must be divisible 32.  If not then round up.
    //uint32_t ba_num_bits = p->num_bits;
    if (p->num_bits % 32 != 0) {
        p->num_bits = (p->num_bits + 31) & -32;
    }

    // construct bitarrays
    p->bitarrays = malloc(p->num_history * sizeof (*p->bitarrays));

    for (uint32_t i = 0; i < p->num_history; i++) {
        p->bitarrays[i] = malloc(sizeof(*p->bitarrays[i]));
        bitarray_construct(p->bitarrays[i], p->num_bits);
    }

    // construct helper bitarray
    p->delta_ba = malloc(sizeof (*p->delta_ba));
    bitarray_construct(p->delta_ba, p->num_bits);

    // set initialized flag to true
    p->init_flag = 1;
}

// =============================================================================
// Add Child
// =============================================================================
void page_add_child(struct Page* p, struct Page* child) {
    p->num_children++;
    p->children = realloc(p->children, p->num_children * sizeof(*p->children));
    p->children[p->num_children - 1] = child;
}

// =============================================================================
// Step
// =============================================================================
void page_step(struct Page* p) {
    p->prev = p->curr;
    p->curr += 1;
    if (p->curr > p->num_history - 1) {
        p->curr = 0;
    }

    page_clear_bits(p, 0);
};

// =============================================================================
// Fetch
// =============================================================================
void page_fetch(struct Page* p) {
    p->changed_flag = 0;
    uint32_t parent_word_offset = 0;
    struct BitArray* parent_ba = p->bitarrays[p->curr];

    for (uint32_t c = 0; c < p->num_children; c++) {
        uint32_t i = p->children[c]->curr;
        struct BitArray* child_ba  = p->children[c]->bitarrays[i];

        if (p->children[c]->changed_flag == 1) {
            p->changed_flag = 1;
        }

        bitarray_copy(
            parent_ba,
            child_ba,
            parent_word_offset,
            0, // child word offset
            child_ba->num_words);

        parent_word_offset += child_ba->num_words;
    }
}

// =============================================================================
// Compute Changed
// =============================================================================
void page_compute_changed(struct Page* p) {
    struct BitArray* curr_ba = p->bitarrays[p->curr];
    struct BitArray* prev_ba = p->bitarrays[p->prev];
    bitarray_xor(curr_ba, prev_ba, p->delta_ba);
    p->changed_flag = 0;
    if (bitarray_count(p->delta_ba) > 0) {
        p->changed_flag = 1;
    }
}

// =============================================================================
// Copy Previous BitArray to Current BitArray
// =============================================================================
void page_copy_previous_to_current(struct Page* p) {
    struct BitArray* curr_ba = p->bitarrays[p->curr];
    struct BitArray* prev_ba = p->bitarrays[p->prev];
    bitarray_copy(curr_ba, prev_ba, 0, 0, prev_ba->num_words);
}

// =============================================================================
// Clear Bits
// =============================================================================
void page_clear_bits(struct Page* p, const uint32_t t) {
    uint32_t i = page_idx_(p, t);
    bitarray_clear(p->bitarrays[i]);
}

// =============================================================================
// Set Bit
// =============================================================================
void page_set_bit(struct Page* p, const uint32_t t, const uint32_t bit) {
    uint32_t i = page_idx_(p, t);
    bitarray_set_bit(p->bitarrays[i], bit);
}

// =============================================================================
// Get Bit
// =============================================================================
uint32_t page_get_bit(struct Page* p, const uint32_t t, const uint32_t bit) {
    uint32_t i = page_idx_(p, t);
    return bitarray_get_bit(p->bitarrays[i], bit);
}

// =============================================================================
// Get BitArray
// =============================================================================
struct BitArray* page_get_bitarray(const struct Page* p, const uint32_t t) {
    uint32_t i = page_idx_(p, t);
    return p->bitarrays[i];
}

// =============================================================================
// Get ActArray
// =============================================================================
struct ActArray* page_get_actarray(const struct Page* p, const uint32_t t) {
    uint32_t i = page_idx_(p, t);
    return bitarray_get_actarray(p->bitarrays[i]);
}

// =============================================================================
// Idx
// =============================================================================
int page_idx_(const struct Page* p, const int t) {
    // TODO: figure out a way to return uint32_t instead of an int
    int i = p->curr - t;
    if (i < 0) {
        i += p->num_history;
    }
    return i;
}