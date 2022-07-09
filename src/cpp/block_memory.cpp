// =============================================================================
// block_memory.cpp
// =============================================================================
#include "block_memory.hpp"
#include "utils.hpp"
#include <cassert>
#include <cstring> // for memset
#include <cstdio>
#include <iostream>

using namespace BrainBlocks;

// =============================================================================
// # Initialize
//
// Initializes BlockMemory based on parameters.
// =============================================================================
void BlockMemory::init(
    const uint32_t num_i,   // number of inputs
    const uint32_t num_d,   // number of dendrites
    const uint32_t num_rpd, // number of receptors per dendrite
    const uint8_t perm_thr, // permanence threshold (0 to 99)
    const uint8_t perm_inc, // permanence increment (0 to 99)
    const uint8_t perm_dec, // permanence decrement (0 to 99)
    const double pct_learn) // learning percentage (0.0 to 1.0)
{

    // Check parameters
    assert(num_d > 0);
    assert(num_rpd > 0);
    assert(perm_thr <= PERM_MAX);
    assert(perm_inc <= PERM_MAX);
    assert(perm_dec <= PERM_MAX);
    assert(pct_learn >= 0.0 && pct_learn <= 1.0);

    // Setup parameters
    this->num_i = num_i;
    this->num_d = num_d;
    this->num_rpd = num_rpd;
    this->num_r = num_d * num_rpd;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_learn = pct_learn;

    // Resize data arrays based on parameters
    state.resize(num_d);
    r_addrs.resize(num_r);
    r_perms.resize(num_r);
    lmask.resize(num_rpd);

    // Setup learning mask
    lmask.set_range(0, (uint32_t)(num_rpd * pct_learn));

    // Clear receptor addresses and permanences
    memset(r_addrs.data(), 0, r_addrs.size() * sizeof(r_addrs[0]));
    memset(r_perms.data(), 0, r_perms.size() * sizeof(r_perms[0]));

    init_flag = true;
}

// =============================================================================
// # Initialize (Connections)
//
// Initialize BlockMemory using connections based on parameters.
// =============================================================================
void BlockMemory::init_conn(
    const uint32_t num_i,   // number of input bitsi
    const uint32_t num_d,   // number of dendrites
    const uint32_t num_rpd, // number of receptors per dendrite
    const uint8_t perm_thr, // permanence threshold (0 to 99)
    const uint8_t perm_inc, // permanence increment (0 to 99)
    const uint8_t perm_dec, // permanence decrement (0 to 99)
    const double pct_learn) // learning percentage (0.0 to 1.0)
{

    assert(num_i > 0);

    init(num_i, num_d, num_rpd, perm_thr, perm_inc, perm_dec, pct_learn);

    d_conns.resize(num_d);

    for (uint32_t d = 0; d < num_d; d++)
        d_conns[d].resize(num_i);

    init_flag = true;
    conns_flag = true;
}

// =============================================================================
// # Initialize Pooled
//
// Defines a dendrite's receptor addresses and permanences from a random
// subsample of the input BitArray space.
//
// ## Example
//
// BlockMemory memory;
//
// memory.init_pooled(32, 8, 0.8, 0.5, 0.3, 20, 2, 1);
//
// input: {00000000000000000000000000000000}
//         ^ ^^^^  ^ ^ ^^^   ^^ ^^^^  ^ ^^^  pct_pool = 0.65
//
// addrs: {00 02 03 04 05 08 10 12 13 14 18 19 21 22 23 24 27 29 30 31}
// perms: {19 20 19 20 20 19 20 19 19 20 20 20 19 19 20 19 20 20 19 19}
//            ^^    ^^ ^^    ^^       ^^ ^^ ^^       ^^    ^^ ^^ pct_conn = 0.5
// =============================================================================
void BlockMemory::init_pooled(
    const uint32_t num_i,   // number of input bits
    const uint32_t num_d,   // number of dendrites
    const double pct_pool,  // pooling percentage (0.0 to 1.0)
    const double pct_conn,  // initially connected percentage (0.0 to 1.0)
    const double pct_learn, // learning percentage (0.0 to 1.0)
    const uint8_t perm_thr, // permanence threshold (0 to 99)
    const uint8_t perm_inc, // permanence increment (0 to 99)
    const uint8_t perm_dec, // permanence decrement (0 to 99)
    std::mt19937& rng)      // random number generator
{

    // Check parameters
    assert(num_i > 0);
    assert(num_d > 0);
    assert(pct_pool >= 0.0 && pct_pool <= 1.0);
    assert(pct_conn >= 0.0 && pct_conn <= 1.0);
    assert(pct_learn >= 0.0 && pct_learn <= 1.0);
    assert(perm_thr <= PERM_MAX);
    assert(perm_inc <= PERM_MAX);
    assert(perm_dec <= PERM_MAX);

    // Setup parameters
    this->num_i = num_i;
    this->num_d = num_d;
    this->num_rpd = (uint32_t)(num_i * pct_pool);
    this->num_r = num_d * num_rpd;
    this->perm_thr = perm_thr;
    this->perm_inc = perm_inc;
    this->perm_dec = perm_dec;
    this->pct_learn = pct_learn;

    // Resize data arrays based on parameters
    state.resize(num_d);
    r_addrs.resize(num_r);
    r_perms.resize(num_r);
    lmask.resize(num_rpd);

    // Setup learning mask
    lmask.set_range(0, (uint32_t)(num_rpd * pct_learn));

    // Setup data arrays using pooled
    uint32_t num_init = (uint32_t)(num_rpd * pct_conn);
    std::vector<uint32_t> rand_addrs(num_i);

    for (uint32_t i = 0; i < num_i; i++)
        rand_addrs[i] = i;

    // Loop through each dendrite
    for (uint32_t d = 0; d < num_d; d++) {

        uint32_t j = 0;
        uint32_t r_beg = d * num_rpd;
        uint32_t r_end = r_beg + num_rpd;

        utils_shuffle(rand_addrs, num_i, rng);

        // Loop through each receptor on the dendrite
        for (uint32_t r = r_beg; r < r_end; r++) {
            r_addrs[r] = rand_addrs[j];

            if (j < num_init)
                r_perms[r] = perm_thr;
            else
                r_perms[r] = perm_thr - 1;

            j++;
        }
    }

    init_flag = true;
}

// =============================================================================
// # Initialize Pooled (Connections)
//
// Initialize pooled using connections based on parameters
// =============================================================================
void BlockMemory::init_pooled_conn(
    const uint32_t num_i,   // number of input bits
    const uint32_t num_d,   // number of dendrites
    const double pct_pool,  // pooling percentage (0.0 to 1.0)
    const double pct_conn,  // initially connected percentage (0.0 to 1.0)
    const double pct_learn, // learning percentage (0.0 to 1.0)
    const uint8_t perm_thr, // permanence threshold (0 to 99)
    const uint8_t perm_inc, // permanence increment (0 to 99)
    const uint8_t perm_dec, // permanence decrement (0 to 99)
    std::mt19937& rng)
{

    init_pooled(num_i, num_d, pct_pool, pct_conn, pct_learn, perm_thr, perm_inc,
                perm_dec, rng);

    d_conns.resize(num_d);

    for (uint32_t d = 0; d < num_d; d++) {
        d_conns[d].resize(num_i);
        update_conns(d);
    }

    init_flag = true;
    conns_flag = true;
}

// =============================================================================
// # Save
//
// Saves memories.
// =============================================================================
void BlockMemory::save(FILE* fptr) {

    std::fwrite(r_addrs.data(), sizeof(r_addrs[0]), r_addrs.size(), fptr);
    std::fwrite(r_perms.data(), sizeof(r_perms[0]), r_perms.size(), fptr);

    if (conns_flag==true) {
        for (int i=0; i<d_conns.size(); i++)  {
            d_conns[i].save(fptr);
        }
    }
}

// =============================================================================
// # Load
//
// Loads memories.
// =============================================================================
void BlockMemory::load(FILE* fptr) {

    std::fread(r_addrs.data(), sizeof(r_addrs[0]), r_addrs.size(), fptr);
    std::fread(r_perms.data(), sizeof(r_perms[0]), r_perms.size(), fptr);

    if (conns_flag==true) {
        for (int i=0; i<d_conns.size(); i++)  {
            d_conns[i].load(fptr);
        }
    }
}

// =============================================================================
// # Clear
//
// Clears state.
// =============================================================================
void BlockMemory::clear() {

    state.clear_all();
}

// =============================================================================
// # Memory Usage
//
// Returns an estimate of the number of bytes used.
// =============================================================================
uint32_t BlockMemory::memory_usage() {

    assert(init_flag);

    uint32_t bytes = 0;

    bytes += state.memory_usage();
    bytes += sizeof(init_flag);
    bytes += sizeof(conns_flag);
    bytes += sizeof(num_d);
    bytes += sizeof(num_rpd);
    bytes += sizeof(num_r);
    bytes += sizeof(perm_thr);
    bytes += sizeof(perm_inc);
    bytes += sizeof(perm_dec);
    bytes += sizeof(pct_learn);
    bytes += (sizeof(r_addrs[0]) * num_r);
    bytes += (sizeof(r_perms[0]) * num_r);
    bytes += lmask.memory_usage();

    if (conns_flag)
        bytes += (d_conns[0].memory_usage() * num_d);

    return bytes;
}

// =============================================================================
// # Overlap
//
// Computes a particular dendrite's overlap value by comparing the input
// BitArray with the dendrite's receptors.
//
// The overlap score is incremented if these conditions are met:
//
// - Connected: The receptor permanence is >= the permanence threshold
// - Active: The input bit at the receptor address is 1
//
// ## Example
//
// overlap(d, input);
//
// perm_thr: 20
//
// addrs[d]: {00    02 03 04 05       08    10    12 13 14   }
// perms[d]: {20    19 19 20 20       19    19    20 19 20   }
//    conns: { 1  0  0  0  1  1  0  0  0  0  0  0  1  0  1  0} perm[r] >= thr
//    input: { 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0}
//    overs: { 1  0  0  0  1  1  0  0  0  0  0  0  0  0  0  0} conns & input
//
//  overlap: 3
// =============================================================================
uint32_t BlockMemory::overlap(const uint32_t d, BitArray& input) {

    assert(init_flag);
    assert(d < num_d);

    uint32_t overlap = 0;
    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    // For each receptor on the dendrite
    for (uint32_t r = r_beg; r < r_end; r++) {

        // If receptor is connected and it's connected bit is active
        // Then increment overlap score
        if (r_perms[r] >= perm_thr && input.get_bit(r_addrs[r]))
            overlap++;
    }

    return overlap;
}

// =============================================================================
// # Overlap (Connections)
//
// See overlap function for description.
// =============================================================================
uint32_t BlockMemory::overlap_conn(const uint32_t d, BitArray& input) {

    assert(init_flag);
    assert(conns_flag);
    assert(d < num_d);

    return d_conns[d].num_similar(input);
}

// =============================================================================
// # Learn
//
// Updates a particular dendrite's receptor permanences based on the input
// BitArray.  Receptors attached to active bits have their permanence value
// incremented while receptors attached to inactive bits have their permanence
// value decremented.  Permanence values that fall below the permanence
// threshold are no longer connected.  Usually only a subset of receptors are
// chosen to adapt.
//
// ## Summary of algorithm
//
// - Learn Mask: Only update the receptor if its learning mask bit is 1
// - Active: Increment permanence if input bit at the receptor address is 1
// - Inactive: Decrement permanence if input bit at the receptor address is 0
// - Minumum permanence value is 0
// - Maximum permanence value is 99
//
// ## Example
//
// learn(d, input);
//
// perm_inc: 2
// perm_dec: 1
//
//    input: { 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0}
// perms[d]: {20    19 19 20 20       19    19    20 19 20   } before
// perms[d]: {22    21 21 22 22       18    18    19 18 19   } after
// =============================================================================
void BlockMemory::learn(
    const uint32_t d,
    BitArray& input,
    std::mt19937& rng)
{

    assert(init_flag);
    assert(d < num_d);

    // Shuffle the learning mask
    if (pct_learn < 1.0)
        lmask.random_shuffle(rng);

    // Get dendrite's receptor boundaries
    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;
    uint32_t l = 0;

    // Loop through each receptor
    for (uint32_t r = r_beg; r < r_end; r++) {

        // If learning mask is set
        if (lmask.get_bit(l++)) {

            // Increment permanence if receptor's input is active
            if (input.get_bit(r_addrs[r]) > 0)
                r_perms[r] = utils_min(r_perms[r] + perm_inc, PERM_MAX);

            // Decrement permanence if receptor's input is inactive
            else
                r_perms[r] = utils_max(r_perms[r] - perm_dec, PERM_MIN);
        }
    }
}

// =============================================================================
// # Learn (Connections)
//
// See learn function for description.
// =============================================================================
void BlockMemory::learn_conn(
    const uint32_t d,
    BitArray& input,
    std::mt19937& rng)
{

    assert(init_flag);
    assert(conns_flag);
    assert(d < num_d);

    learn(d, input, rng);
    update_conns(d);
}

// =============================================================================
// # Learn and Move
//
// Updates a particular dendrite's receptor addresses and permanences based on
// the input BitArray.  This function is similar to the learn() function except
// when a permanence value reaches zero the receptor is moved to an unused
// bit in the input BitArray.  Moving a receptor has the advantage of maximizing
// the use of receptors with the input BitArray.  When a receptor is moved the
// permanence value is set to the permanence threshold value.
//
// ## Example
//
// learn_move(d, input);
//
// perm_inc: 2
// perm_dec: 1
//
//    input: { 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0}
// addrs[d]: {00    02 03 04 05       08    10    12 13 14   } before
// perms[d]: {20    19 19 20 20       00    19    20 19 20   } before
// addrs[d]: {00    02 03 04 05 06          10    12 13 14   } after
// perms[d]: {22    21 21 22 22 20          18    19 18 19   } after
//                               ^     |
//                               +-----+
//                            receptor moved
// =============================================================================
void BlockMemory::learn_move(
    const uint32_t d,
    BitArray& input,
    std::mt19937& rng)
{

    assert(init_flag);
    assert(d < num_d);

    uint32_t l = 0;
    uint32_t next_addr = 0;

    // Shuffle the learning mask
    if (pct_learn < 1.0)
        lmask.random_shuffle(rng);

    // Get dendrite's receptor boundaries
    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    // FIXME: available input bits here are selected from already connected receptors.
    // FIXME: should sample over unconnected input space

    // Get available input bits
    BitArray available = input;

    // clear bits we are already have receptors
    for (uint32_t r = r_beg; r < r_end; r++) {
        if (r_perms[r] > 0)
            available.clear_bit(r_addrs[r]);
    }

    // Loop through each receptor
    for (uint32_t r = r_beg; r < r_end; r++) {

        // If learning mask is set
        if (lmask.get_bit(l++)) {

            // If receptor permanence is above zero then perform normal learning
            if (r_perms[r] > 0) {

                // Increment permanence if receptor's input is active
                if (input.get_bit(r_addrs[r]) > 0)
                    r_perms[r] = utils_min(r_perms[r] + perm_inc, PERM_MAX);

                // Decrement permanence if receptor's input is inactive
                else
                    r_perms[r] = utils_max(r_perms[r] - perm_dec, PERM_MIN);
            }

            // If receptor permanence is below zero then move address to an
            // unused active input bit
            else {
                bool pass = available.find_next_set_bit(next_addr, &next_addr);

                if (!pass)
                    continue;

                r_addrs[r] = next_addr;
                r_perms[r] = perm_thr;
                available.clear_bit(next_addr);
            }
        }
    }
}

// =============================================================================
// # Learn and Move (Connections)
//
// see learn_move function for description.
// =============================================================================
void BlockMemory::learn_move_conn(
    const uint32_t d,
    BitArray& input,
    std::mt19937& rng)
{

    assert(init_flag);
    assert(conns_flag);
    assert(d < num_d);

    learn_move(d, input, rng);
    update_conns(d);
}

// =============================================================================
// # Punish Dendrite
//
// Updates a particular dendrite's receptor permanences based on the input
// BitArray. This function simply decrements receptor permanences if they are
// attached to active input bits.
//
// ## Example
//
// punish(d, input);
//
// perm_inc: 2
//
//    input: { 1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0}
// perms[d]: {20    19 19 20 20       19    19    20 19 20   } before
// perms[d]: {18    17 17 18 18       19    19    29 19 20   } after
// =============================================================================
void BlockMemory::punish(
    const uint32_t d,
    BitArray& input,
    std::mt19937& rng)
{

    assert(init_flag);
    assert(d < num_d);

    // Shuffle the learning mask
    if (pct_learn < 1.0)
        lmask.random_shuffle(rng);

    // Get dendrite's receptor boundaries
    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;
    uint32_t l = 0;

    // Loop through each receptor
    for (uint32_t r = r_beg; r < r_end; r++) {

        // If receptor learning mask is set
        if (lmask.get_bit(l++)) {

            // Decrement permanence by perm_inc if receptor's input is active
            if (input.get_bit(r_addrs[r]) > 0)
                r_perms[r] = utils_max(r_perms[r] - perm_inc, PERM_MIN);
        }
    }
}

// =============================================================================
// # Punish (Connections)
//
// See punish function for description.
// =============================================================================
void BlockMemory::punish_conn(
    const uint32_t d,
    BitArray& input,
    std::mt19937& rng)
{

    assert(init_flag);
    assert(conns_flag);
    assert(d < num_d);

    punish(d, input, rng);
    update_conns(d);
}

// =============================================================================
// # Print Receptor Addresses Dendrite
//
// Prints the receptor addresses on a particular dendrite.
// =============================================================================
void BlockMemory::print_addrs(const uint32_t d) {

    assert(init_flag);
    assert(d < num_d);

    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    std::cout << "{";

    for (uint32_t r = r_beg; r < r_end; r++) {
        std::cout << r_addrs[r];

        if (r < r_end - 1)
            std::cout << ", ";
    }

    std::cout << "}" << std::endl;
}

// =============================================================================
// # Print Receptor Permanences Dendrite
//
// Prints the receptor permanences on a particular dendrite.
// =============================================================================
void BlockMemory::print_perms(const uint32_t d) {

    assert(init_flag);
    assert(d < num_d);

    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    std::cout << "{";

    for (uint32_t r = r_beg; r < r_end; r++) {
        std::cout << (uint32_t)r_perms[r];

        if (r < r_end - 1)
            std::cout << ", ";
    }

    std::cout << "}" << std::endl;
}

// =============================================================================
// # Print Receptor Connections Dendrite
//
// Prints the receptor connections on a particular dendrite.
// =============================================================================
void BlockMemory::print_conns(const uint32_t d) {

    assert(init_flag);
    assert(conns_flag);
    assert(d < num_d);

    d_conns[d].print_bits();
}

// =============================================================================
// # Get Dendrite Addresses
//
// Returns the receptor addresses of a particular dendrite.
// =============================================================================
std::vector<uint32_t> BlockMemory::addrs(const uint32_t d) {

    assert(init_flag);
    assert(d < num_d);

    uint32_t i = 0;
    std::vector<uint32_t> addrs(num_rpd);

    // Get dendrite's receptor boundaries
    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    // For each receptor on the dendrite
    for (uint32_t r = r_beg; r < r_end; r++)
        addrs[i++] = r_addrs[r];

    return addrs;
}

// =============================================================================
// # Get Dendrite Permanences
//
// Returns the receptor permanences of a particular dendrite.
// =============================================================================
std::vector<uint8_t> BlockMemory::perms(const uint32_t d) {

    assert(init_flag);
    assert(d < num_d);

    uint32_t i = 0;
    std::vector<uint8_t> perms(num_rpd);

    // Get dendrite's receptor boundaries
    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    // For each receptor on the dendrite
    for (uint32_t r = r_beg; r < r_end; r++)
        perms[i++] = r_perms[r];

    return perms;
}

// =============================================================================
// # Get Dendrite Connections
//
// Returns the receptor connections of a particular dendrite.
// =============================================================================
std::vector<uint8_t> BlockMemory::conns(const uint32_t d) {

    assert(init_flag);
    assert(d < num_d);

    uint32_t i = 0;
    std::vector<uint8_t> conns(num_i);

    // Zero conns vector
    memset(conns.data(), 0, conns.size() * sizeof(conns[0]));

    // Get dendrite's receptor boundaries
    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    // For each receptor on the dendrite
    for (uint32_t r = r_beg; r < r_end; r++) {
        if (r_perms[r] >= perm_thr)
            conns[r_addrs[r]] = 1;
    }

    return conns;
}

// =============================================================================
// # Update Connections Dendrite
//
// Updates the receptor connections on a particular dendrite.
// =============================================================================
void BlockMemory::update_conns(const uint32_t d) {

    assert(init_flag);
    assert(d < num_d);

    d_conns[d].clear_all();

    uint32_t r_beg = d * num_rpd;
    uint32_t r_end = r_beg + num_rpd;

    for (uint32_t r = r_beg; r < r_end; r++) {
        if (r_perms[r] >= perm_thr)
            d_conns[d].set_bit(r_addrs[r]);
    }
}
