#include "coincidence_set.hpp"
#include "utils.hpp"
#include <iostream>

// =============================================================================
// Resize
// =============================================================================
void CoincidenceSet::resize(const uint32_t num_r) {
    addrs.resize(num_r);
    perms.resize(num_r);
    
    for (uint32_t r = 0; r < addrs.size(); r++) {
        addrs[r] = 0;
        perms[r] = 0;
    }
}

// =============================================================================
// Initialize Pool
// =============================================================================
void CoincidenceSet::initialize_pool(
        const uint32_t num_r,     // number of receptors
        const uint32_t num_i,     // number of input bits
        const uint32_t num_conn,  // number of initially connected receptors
        const uint8_t perm_thr) { // permanence threshold

    // error check
    if (num_conn > num_r) {
        std::cout << "Error in CoincidenceSet(): num_conn > num_r" << std::endl;
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        std::cout << "Error in CoincidenceSet(): perm_thr > PERM_MAX" << std::endl;
        exit(1);
    }

    // initialize variables
    addrs.resize(num_r);
    perms.resize(num_r);

    // shuffle temporary random address array
    std::vector<uint32_t> rand_addrs(num_i);

    for (uint32_t i = 0; i < num_i; i++) {
        rand_addrs[i] = i;
    }

    utils_shuffle(rand_addrs, num_i);

    // randomize address and permanence arrays
    uint32_t j = 0;

    for (uint32_t r = 0; r < addrs.size(); r++) {
        addrs[r] = rand_addrs[r];
        
        if (j++ < num_conn) {
            perms[r] = perm_thr;
        }
        else {
            perms[r] = perm_thr - 1;
        }
    }
}

// =============================================================================
// Overlap
// =============================================================================
uint32_t CoincidenceSet::overlap(BitArray& input_ba, const uint8_t perm_thr) {
    BitArray conns_ba(input_ba.get_num_bits());
    
    // loop through each receptor
    for (uint32_t r = 0; r < addrs.size(); r++) {

        // if receptor permanence is above the threshold then set the connection
        if (perms[r] >= perm_thr) {
            conns_ba.set_bit(addrs[r], 1);
        }
    }

    BitArray overlap_ba = conns_ba & input_ba;
    return overlap_ba.count();
}

// =============================================================================
// Learn
// =============================================================================
void CoincidenceSet::learn(
        BitArray& input_ba,
        BitArray& lmask_ba,
        const uint8_t perm_inc,
        const uint8_t perm_dec) {

    // loop through each receptor
    for (uint32_t r = 0; r < addrs.size(); r++) {

        // if receptor learning mask is set
        if (lmask_ba.get_bit(r) > 0) {

            // increment permanence if receptor's input is active
            if (input_ba.get_bit(addrs[r]) > 0) {
                perms[r] = MIN(perms[r] + perm_inc, PERM_MAX);
            }

            // decrement permanence if receptor's input is inactive
            else {
                perms[r] = MAX(perms[r] - perm_dec, PERM_MIN);
            }
        }
    }
}

// =============================================================================
// Learn (Move)
// =============================================================================
void CoincidenceSet::learn_move(
        BitArray& input_ba,
        BitArray& lmask_ba,
        const uint8_t perm_inc,
        const uint8_t perm_dec) {

    // get active bits
    std::vector<uint32_t> acts = input_ba.get_acts();

    // loop through each receptor
    for (uint32_t r = 0; r < addrs.size(); r++) {

        // if receptor learning mask is set
        if (lmask_ba.get_bit(r) > 0) {

            // if receptor permanence is above zero then perform normal learning
            if (perms[r] > 0) {

                // increment permanence if receptor's input is active
                if (input_ba.get_bit(addrs[r]) > 0) {
                    perms[r] = MIN(perms[r] + perm_inc, PERM_MAX);
                }

                // decrement permanence if receptor's input is inactive
                else {
                    perms[r] = MAX(perms[r] - perm_dec, PERM_MIN);
                }
            }

            // if permanence is zero then move receptor
            // TODO: Need to optimize this... maybe use bitarrays?
            else if (acts.size() > 0) {
                for (uint32_t j = 0; j < acts.size(); j++) {
                    bool is_available = true;

                    // check if the input address is already used on the CoincidenceSet
                    for (uint32_t k = 0; k <= addrs.size(); k++) {
                        if (acts[j] == addrs[k] && perms[k] > 0) {
                            is_available = false;
                            break;
                        }
                    }

                    // move the receptor if it is available
                    if (is_available) {
                        addrs[r] = acts[j];
                        perms[r] = perm_inc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Punish
// =============================================================================
void CoincidenceSet::punish(
            BitArray& input_ba,
            BitArray& lmask_ba,
            const uint8_t perm_inc) {

    // loop through each receptor
    for (uint32_t r = 0; r < addrs.size(); r++) {

        // if receptor learning mask is set
        if (lmask_ba.get_bit(r) > 0) {

            // decrement permanence if receptor's input is active
            if (input_ba.get_bit(addrs[r]) > 0) {
                perms[r] = MAX(perms[r] - perm_inc, PERM_MIN);
            }
        }
    }
}

// =============================================================================
// Print Addresses
// =============================================================================
void CoincidenceSet::print_addrs() {
    std::cout << "{";
    for (uint32_t r = 0; r < addrs.size(); r++) {
        std::cout << addrs[r];
        if (r < addrs.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "}" << std::endl;
}

// =============================================================================
// Print Permanences
// =============================================================================
void CoincidenceSet::print_perms() {
    std::cout << "{";
    for (uint32_t r = 0; r < addrs.size(); r++) {
        std::cout << (uint32_t)perms[r];
        if (r < addrs.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "}" << std::endl;
}