#include "page.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
Page::Page() {
    num_bits = 0;
    num_bitarrays = 2;
    curr = 0;
    prev = 1;
    changed_flag = true;
    init_flag = false;
}

// =============================================================================
// Initialize
// =============================================================================
void Page::initialize() {

    // update num_bits from children
    for (uint32_t c = 0; c < children.size(); c++) {

        // initialize child if necessary
        if (!children[c]->is_initialized()) {
            children[c]->initialize();
        }
        
        num_bits += children[c]->get_num_bits();
    }

    // TODO: might need better way of handling this.  User probably forgot
    // to attach blocks to this block and it's left hanging and alone
    // Page num_bits must be greater than 0
    if (num_bits == 0) {
        std::cout << "Error in Page::initialize(): num_bits must be greater than 0.  Make sure your block input page has children or you set num_bits." << std::endl;
        exit(1);
    }

    // BitArrays num_bits must be divisible 32.  If not then round up.
    if (num_bits % 32 != 0) {
        num_bits = (num_bits + 31) & -32;
    }

    // initialize bitarrays
    bitarrays.resize(num_bitarrays);
    for (uint32_t i = 0; i < bitarrays.size(); i++) {
        bitarrays[i].resize(num_bits);
    }

    // set initialized flag to true
    init_flag = true;
}

// =============================================================================
// Add Child
// =============================================================================
void Page::add_child(Page& child) {
    children.push_back(&child);
}

// =============================================================================
// Step
// =============================================================================
void Page::step() {
    if (init_flag == false) {
        std::cout << "Error in Page::step(): page has not been initialized." << std::endl;
        exit(1);
    }

    prev = curr;
    curr += 1;
    if (curr > (uint32_t)bitarrays.size() - 1) {
        curr = 0;
    }

    bitarrays[curr].clear_bits();
};

// =============================================================================
// Fetch
// =============================================================================
void Page::fetch() {
    if (init_flag == false) {
        std::cout << "Error in Page::fetch(): page has not been initialized." << std::endl;
        exit(1);
    } 

    changed_flag = false;
    uint32_t parent_word_offset = 0;
    BitArray* parent_ba = &bitarrays[curr];

    for (uint32_t c = 0; c < children.size(); c++) {
        uint32_t i = children[c]->curr;
        BitArray* child_ba  = &children[c]->bitarrays[i];

        if (children[c]->changed_flag == true) {
            changed_flag = true;
        }

        uint32_t child_num_words = child_ba->get_num_words();
        bitarray_copy(*parent_ba, *child_ba, parent_word_offset, 0, child_num_words);
        parent_word_offset += child_num_words;
    }
}

// =============================================================================
// Compute Changed
// =============================================================================
void Page::compute_changed() {
    changed_flag = false;
    BitArray delta_ba = bitarrays[curr] ^ bitarrays[prev];
    if (delta_ba.count() > 0) {
        changed_flag = true;
    }
}

// =============================================================================
// Copy Previous BitArray to Current BitArray
// =============================================================================
void Page::copy_previous_to_current() {
    bitarray_copy(bitarrays[curr], bitarrays[prev], 0, 0, bitarrays[prev].get_num_words());
}

/*
// =============================================================================
// Get Child
// =============================================================================
Page* Page::get_child(const uint32_t child_idx) {
    if (child_idx > children.size()) {
        std::cout << "Error in Page::get_child(): child_index out of range." << std::endl;
        exit(1);
    }

    return children[child_idx];
}
*/

// =============================================================================
// Get BitArray (Operator)
// =============================================================================
BitArray& Page::operator[](const uint32_t t) {
    if (init_flag == false) {
        std::cout << "Error in Page::operator(): This Page has not been initialized" << std::endl;
        exit(1);
    }

    uint32_t i = get_index(t);
    return bitarrays[i];
}

// =============================================================================
// Print Information
// =============================================================================
void Page::print_info() {
    std::cout << "{"<< std::endl;
    std::cout << "    \"object\": Page," << std::endl;
    std::cout << "    \"address\": 0x" << this << "," << std::endl;
    std::cout << "    \"init_flag\": " << init_flag << "," << std::endl;

    size_t num_children = children.size();
    std::cout << "    \"children:\": [";
    if (num_children > 0) {
        std::cout << std::endl;
        for (uint32_t i = 0; i < children.size(); i++) {
            std::cout << "        0x" << children[i] << "," << std::endl;
        }
        std::cout << "    ";
    }
    std::cout << "]," << std::endl;

    size_t num_bitarrays = bitarrays.size();
    std::cout << "    \"bitarrays:\": [";
    if (num_bitarrays > 0) {
        std::cout << std::endl;
        for (uint32_t i = 0; i < bitarrays.size(); i++) {
            std::cout << "        0x" << &bitarrays[i] << "," << std::endl;
        }
        std::cout << "    ";
    }
    std::cout << "]," << std::endl;

    std::cout << "}," << std::endl;
}

// =============================================================================
// Get Index
// =============================================================================
int Page::get_index(const int t) {
    // TODO: figure out a way to return uint32_t instead of an int
    int i = curr - t;
    if (i < 0) {
        i += (int)bitarrays.size();
    }
    return i;
}