#ifndef TEST_BLANK_BLOCK_H
#define TEST_BLANK_BLOCK_H

#include "blank_block.h"

#include "helper.h"
#include "utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

void test_blank_block() {
    printf("================================================================================\n");
    printf("Test BlankBlock\n");
    printf("================================================================================\n");
    printf("\n");
    
    utils_seed(0);

    struct BlankBlock bb;

    printf("blank_block_construct()\n");
    blank_block_construct(&bb, 32);
    printf("bb output=");
    page_print_bits(bb.output, CURR);
    printf("\n");

    printf("printing parameters:\n");
    blank_block_print_parameters(&bb);
    printf("\n");

    printf("blank_block_destruct()\n");
    blank_block_destruct(&bb);
}

#endif