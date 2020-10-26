#ifndef COINCIDENCE_SET_HPP
#define COINCIDENCE_SET_HPP

#include "bitarray.hpp"
#include <stdint.h>

#define PERM_MIN 0
#define PERM_MAX 99

struct CoincidenceSet {
    uint32_t num_r;   // number of input receptors
    uint32_t state;   // output binary state // TODO: make int8_t
    uint32_t overlap; // overlap score
    uint32_t templap; // temporary overlap score TODO: remove and update pooler and classifier activate funcs
    uint32_t* addrs;  // receptor addresses
    int8_t* perms;    // receptor permanences
    struct BitArray* connections_ba;
    struct BitArray* activeconns_ba;
};

void coincidence_set_construct(
    struct CoincidenceSet* cs,
    const uint32_t num_r);

void coincidence_set_construct_pooled(
    struct CoincidenceSet* cs,
    const uint32_t num_i,     // number of input bits
    const uint32_t num_r,     // number of receptors
    const uint32_t num_conn,  // number of initially connected receptors
    const uint32_t perm_thr); // permanence threshold

void coincidence_set_destruct(
    struct CoincidenceSet* cs);

void coincidence_set_overlap(
    struct CoincidenceSet* cs,
    const struct BitArray* input_ba);

void coincidence_set_learn(
    struct CoincidenceSet* cs,
    const struct BitArray* input_ba,
    const uint32_t* learn_mask,
    const uint32_t perm_inc,
    const uint32_t perm_dec);

void coincidence_set_learn_move(
    struct CoincidenceSet* cs,
    const struct BitArray* input_ba,
    const struct ActArray* input_aa,
    const uint32_t perm_inc,
    const uint32_t perm_dec);

void coincidence_set_punish(
    struct CoincidenceSet* cs,
    const struct BitArray* input_ba,
    const uint32_t* learn_mask,
    const uint32_t perm_inc);

void coincidence_set_update_connections(
    struct CoincidenceSet* cs,
    const uint32_t perm_thr);

struct BitArray* coincidence_set_get_connections(struct CoincidenceSet* cs);

#endif