#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <vector>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void utils_seed(const uint32_t seed);
uint32_t utils_rand_uint(uint32_t min, uint32_t max);
void utils_shuffle(std::vector<uint32_t>& A, uint32_t n);

#endif