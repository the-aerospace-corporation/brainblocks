#ifndef BLANK_BLOCK_H
#define BLANK_BLOCK_H

#include "page.h"
#include <stdint.h>

struct BlankBlock {
    uint32_t num_s; // number of statelets
    struct Page* output;
};

void blank_block_construct(struct BlankBlock* b, const uint32_t num_s);
void blank_block_destruct(struct BlankBlock* b);
void blank_block_clear(struct BlankBlock* b);

#endif