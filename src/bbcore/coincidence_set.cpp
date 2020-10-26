#include "coincidence_set.hpp"
#include "utils.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

// =============================================================================
// Constructor
// =============================================================================
void coincidence_set_construct(
        struct CoincidenceSet* cs,
        const uint32_t num_r) {

    // initialize variables
    cs->num_r = num_r;
    cs->state = 0;
    cs->overlap = 0;
    cs->templap = 0;
    cs->addrs = (uint32_t*)calloc(cs->num_r, sizeof(*cs->addrs));
    cs->perms = (int8_t*)calloc(cs->num_r, sizeof(*cs->perms));
    cs->connections_ba = NULL;
    cs->activeconns_ba = NULL;
}

// =============================================================================
// Constructor (Pooled)
// =============================================================================
void coincidence_set_construct_pooled(
        struct CoincidenceSet* cs,
        const uint32_t num_i,      // number of input bits
        const uint32_t num_r,      // number of receptors
        const uint32_t num_conn,   // number of initially connected receptors
        const uint32_t perm_thr) { // permanence threshold

    // error check
    if (num_conn > num_r) {
        perror("Error: CoincidenceSet num_conn > num_r");
        exit(1);
    }

    if (perm_thr > PERM_MAX) {
        perror("Error: CoincidenceSet perm_thr > 99");
        exit(1);
    }

    // initialize variables
    cs->num_r = num_r;
    cs->state = 0;
    cs->overlap = 0;
    cs->templap = 0;
    cs->addrs = (uint32_t*)calloc(cs->num_r, sizeof(*cs->addrs));
    cs->perms = (int8_t*)calloc(cs->num_r, sizeof(*cs->perms));
    cs->connections_ba = (BitArray*)malloc(sizeof(*cs->connections_ba));
    cs->activeconns_ba = (BitArray*)malloc(sizeof(*cs->activeconns_ba));

    // construct bitarrays
    bitarray_construct(cs->connections_ba, num_i);
    bitarray_construct(cs->activeconns_ba, num_i);

    // shuffle temporary random address array
    uint32_t* rand_addrs = (uint32_t*)malloc(num_i * sizeof(*rand_addrs));

    for (uint32_t i = 0; i < num_i; i++) {
        rand_addrs[i] = i;
    }

    utils_shuffle(rand_addrs, num_i);

    // randomize address and permanence arrays
    uint32_t j = 0;

    for (uint32_t r = 0; r < cs->num_r; r++) {
        cs->addrs[r] = rand_addrs[r];
        
        if (j++ <= num_conn) {
            cs->perms[r] = perm_thr;
            bitarray_set_bit(cs->connections_ba, cs->addrs[r]);
        }
        else {
            cs->perms[r] = perm_thr - 1;
        }
    }

    free(rand_addrs);
}

// =============================================================================
// Destructor
// =============================================================================
void coincidence_set_destruct(struct CoincidenceSet* cs) {
    // destruct connections bitarray
    if (cs->connections_ba != NULL) {
        bitarray_destruct(cs->connections_ba);
    }

    // destruct activeconns bitarray
    if (cs->activeconns_ba != NULL) {
        bitarray_destruct(cs->activeconns_ba);
    }

    // free pointers
    free(cs->connections_ba);
    free(cs->activeconns_ba);
    free(cs->addrs);
    free(cs->perms);
}

// =============================================================================
// Overlap
// =============================================================================
void coincidence_set_overlap(
        struct CoincidenceSet* cs,
        const struct BitArray* input_ba) {

    // update overlap BitArray through a bitwise and of connections and input
    bitarray_clear(cs->activeconns_ba);
    bitarray_and(cs->connections_ba, input_ba, cs->activeconns_ba);

    // get overlap value by counting the number of active bits
    uint32_t overlap = bitarray_count(cs->activeconns_ba);
    cs->overlap = overlap;
    cs->templap = overlap;
}

// =============================================================================
// Learn
// =============================================================================
// For each receptor:
//   - update only if the receptor has been chosen to update via the learn_mask
//   - increment permanence if receptor's input is active
//   - decrement permanence if receptor's input is inactive
void coincidence_set_learn(
        struct CoincidenceSet* cs,
        const struct BitArray* input_ba,
        const uint32_t* learn_mask,
        const uint32_t perm_inc,   // TODO: change to uint8_t
        const uint32_t perm_dec) { // TODO: change to uint8_t

    // loop through each receptor
    for (uint32_t r = 0; r < cs->num_r; r++) {
        if (learn_mask[r] > 0) {
            if (bitarray_get_bit(input_ba, cs->addrs[r])) {
                cs->perms[r] = MIN(cs->perms[r] + (int8_t)perm_inc, PERM_MAX);
            }
            else {
                cs->perms[r] = MAX(cs->perms[r] - (int8_t)perm_dec, PERM_MIN);
            }
        }
    }
}

// =============================================================================
// Learn (Move)
// =============================================================================
// For each receptor:
//   - update only if the receptor has been chosen to update via the learn_mask
//   - increment permanence if receptor's input is active
//   - decrement permanence if receptor's input is inactive
//   - if permanence is zero move the receptor to an unused active input
void coincidence_set_learn_move(
        struct CoincidenceSet* cs,
        const struct BitArray* input_ba,
        const struct ActArray* input_aa, // TODO: add learn_mask?
        const uint32_t perm_inc,   // TODO: change to uint8_t
        const uint32_t perm_dec) { // TODO: change to uint8_t

    // loop through each receptor
    for (uint32_t r = 0; r < cs->num_r; r++) {
        if (cs->perms[r] > 0) {
            if (bitarray_get_bit(input_ba, cs->addrs[r])) {
                cs->perms[r] = MIN(cs->perms[r] + (int8_t)perm_inc, PERM_MAX);
            }
            else {
                cs->perms[r] = MAX(cs->perms[r] - (int8_t)perm_dec, PERM_MIN);
            }
        }

        // if permanence is zero then move receptor
        // TODO: Need to optimize this... maybe use bitarrays?
        else if (input_aa->num_acts > 0) {
            for (uint32_t j = 0; j < input_aa->num_acts; j++) {
                uint32_t is_available = 1;

                // check if already used on the dendrite
                for (uint32_t k = 0; k <= cs->num_r; k++) {
                    if (input_aa->acts[j] == cs->addrs[k] && cs->perms[k] > 0) {
                        is_available = 0;
                        break;
                    }
                }
                
                if (is_available) {
                    cs->addrs[r] = input_aa->acts[j];
                    cs->perms[r] = perm_inc;
                }
            }
        }
    }
}

// =============================================================================
// Punish
// =============================================================================
// For each receptor:
//   - update only if the receptor has been chosen to update via the learn_mask
//   - decrement permanence if receptor's input is active
void coincidence_set_punish(
        struct CoincidenceSet* cs,
        const struct BitArray* input_ba,
        const uint32_t* learn_mask,
        const uint32_t perm_inc) { // TODO: change to uint8_t

    // loop through each receptor
    for (uint32_t r = 0; r < cs->num_r; r++) {
        if (learn_mask[r] > 0) {
            if (bitarray_get_bit(input_ba, cs->addrs[r])) {
                cs->perms[r] = MAX(cs->perms[r] - (int8_t)perm_inc, PERM_MIN);
            }
        }
    }
}

// =============================================================================
// Update Connections BitArray
// =============================================================================
// For each receptor set the connections bitarray bit to active if the
// receptor permanence is above the permanence theshold
void coincidence_set_update_connections(
        struct CoincidenceSet* cs,
        const uint32_t perm_thr) { // TODO: change to uint8_t

    bitarray_clear(cs->connections_ba);

    // loop through each receptor
    for (uint32_t r = 0; r < cs->num_r; r++) {
        if (cs->perms[r] >= (int8_t)perm_thr) {
            bitarray_set_bit(cs->connections_ba, cs->addrs[r]);
        }
    }
}

// =============================================================================
// Get Connections
// =============================================================================
struct BitArray* coincidence_set_get_connections(struct CoincidenceSet* cs) {
    coincidence_set_update_connections(cs, 20); // TODO: put perm_thr in coincidence_set
    return cs->connections_ba;
}