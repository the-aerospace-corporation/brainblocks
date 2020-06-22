#include "blank_block.h"

#include <stdlib.h>

// =============================================================================
// Constructor
// =============================================================================
void blank_block_construct(struct BlankBlock* b, const uint32_t num_s) {

    // initialize variables
    b->num_s = num_s;
    b->output = malloc(sizeof(*b->output));
    page_construct(b->output, 2, b->num_s);
    page_initialize(b->output);
}

// =============================================================================
// Destructor
// =============================================================================
void blank_block_destruct(struct BlankBlock* b) {
    page_destruct(b->output);
    free(b->output);
}