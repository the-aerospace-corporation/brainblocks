#include "page.hpp"
#include <iostream>

// =============================================================================
// Constructor
// =============================================================================
Page::Page(const uint32_t num_history, const uint32_t num_bits) {
    this->num_bits = num_bits;
    curr = 0;
    prev = 1;
    changed_flag = true;
    init_flag = false;
    bitarrays.resize(num_history);
}

// =============================================================================
// Destructor
// =============================================================================
Page::~Page() {
    for (uint32_t i = 0; i < bitarrays.size(); i++) {
        delete bitarrays[i];
    }
}

// =============================================================================
// Initialize
// =============================================================================
void Page::initialize() {

    // update num_bits from children
    for (uint32_t c = 0; c < children.size(); c++) {
        Page* child = children[c];
        
        if (!child->is_initialized()) {
            child->initialize();
        }
        
        num_bits += child->get_num_bits();
    }

    // TODO: might need better way of handling this.  User probably forgot
    // to attach blocks to this block and it's left hanging and alone
    // Page num_bits must be greater than 0
    if (num_bits == 0) {
        printf("Error in page_initialize: num_bits must be greater than 0.  Make sure your block input page has children.");
        exit(1);
    }

    // BitArrays num_bits must be divisible 32.  If not then round up.
    //uint32_t ba_num_bits = num_bits;
    if (num_bits % 32 != 0) {
        num_bits = (num_bits + 31) & -32;
    }

    // initialize bitarrays
    for (uint32_t i = 0; i < bitarrays.size(); i++) {
        bitarrays[i] = new BitArray(num_bits);
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
// Get Child
// =============================================================================
Page* Page::get_child(const uint32_t child_idx) {
    if (child_idx > children.size()) {
        std::cout << "Error in Page::get_child(): child_index out of range." << std::endl;
        exit(1);
    }

    return children[child_idx];
}

// =============================================================================
// Clear Bit
// =============================================================================
void Page::clear_bit(const uint32_t t, const uint32_t idx) {
    if (init_flag == false) {
        std::cout << "Error in Page::clear_bit(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    bitarrays[i]->clear_bit(idx);
}

// =============================================================================
// Set Bit
// =============================================================================
void Page::set_bit(const uint32_t t, const uint32_t idx) {
    if (init_flag == false) {
        std::cout << "Error in Page::set_bit(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    bitarrays[i]->set_bit(idx);
}

// =============================================================================
// Get Bit
// =============================================================================
uint32_t Page::get_bit(const uint32_t t, const uint32_t idx) {
    if (init_flag == false) {
        std::cout << "Error in Page::get_bit(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    return bitarrays[i]->get_bit(idx);
}

// =============================================================================
// Clear Bits
// =============================================================================
void Page::clear_bits(const uint32_t t) {
    if (init_flag == false) {
        std::cout << "Error in Page::clear_bits(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    bitarrays[i]->clear_bits();
}

// =============================================================================
// Set Bits
// =============================================================================
void Page::set_bits(const uint32_t t, std::vector<uint8_t>& bits) {
    if (init_flag == false) {
        std::cout << "Error in Page::set_bits(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    bitarrays[i]->set_bits(bits);
}

// =============================================================================
// Set Acts
// =============================================================================
void Page::set_acts(const uint32_t t, std::vector<uint32_t>& acts) {
    if (init_flag == false) {
        std::cout << "Error in Page::set_acts(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    bitarrays[i]->set_acts(acts);
}

// =============================================================================
// Get Bits
// =============================================================================
std::vector<uint8_t> Page::get_bits(const uint32_t t) {
    if (init_flag == false) {
        std::cout << "Error in Page::get_bits(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    return bitarrays[i]->get_bits();
}

// =============================================================================
// Get Acts
// =============================================================================
std::vector<uint32_t> Page::get_acts(const uint32_t t) {
    if (init_flag == false) {
        std::cout << "Error in Page::get_acts(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    return bitarrays[i]->get_acts();
}

// =============================================================================
// Print Bits
// =============================================================================
void Page::print_bits(const uint32_t t) {
    if (init_flag == false) {
        std::cout << "Error in Page::print_bits(): page has not been initialized." << std::endl;
        exit(1);
    }
    uint32_t i = get_index(t);
    bitarrays[i]->print_bits();
}

// =============================================================================
// Print Acts
// =============================================================================
void Page::print_acts(const uint32_t t) {
    if (init_flag == false) {
        std::cout << "Error in Page::print_acts(): page has not been initialized." << std::endl;
        exit(1);
    }    
    uint32_t i = get_index(t);
    bitarrays[i]->print_acts();
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

    clear_bits(0);
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
    BitArray* parent_ba = bitarrays[curr];

    for (uint32_t c = 0; c < children.size(); c++) {
        uint32_t i = children[c]->curr;
        BitArray* child_ba  = children[c]->bitarrays[i];

        if (children[c]->changed_flag == true) {
            changed_flag = true;
        }

        uint32_t child_num_words = child_ba->get_num_words();
        bitarray_copy(parent_ba, child_ba, parent_word_offset, 0, child_num_words);
        parent_word_offset += child_num_words;
    }
}

// =============================================================================
// Compute Changed
// =============================================================================
void Page::compute_changed() {
    changed_flag = false;
    BitArray delta_ba = *bitarrays[curr] ^ *bitarrays[prev];
    if (delta_ba.count() > 0) {
        changed_flag = true;
    }
}

// =============================================================================
// Copy Previous BitArray to Current BitArray
// =============================================================================
void Page::copy_previous_to_current() {
    bitarray_copy(bitarrays[curr], bitarrays[prev], 0, 0, bitarrays[prev]->num_words);
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