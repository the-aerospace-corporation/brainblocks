// =============================================================================
// utils.hpp
// =============================================================================
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>

#define utils_min(a,b) (((a)<(b))?(a):(b))
#define utils_max(a,b) (((a)>(b))?(a):(b))

namespace BrainBlocks {

// =============================================================================
// Utils Rand Uint
// =============================================================================
inline uint32_t utils_rand_uint(
    const uint32_t min,
    const uint32_t max,
    std::mt19937& rng)
{

    return (rng() % (max - min + 1)) + min;
}

// =============================================================================
// Utils Shuffle
// =============================================================================
inline void utils_shuffle(
    std::vector<uint32_t>& A,
    const uint32_t n,
    std::mt19937& rng)
{

    for (int i = n - 1; i >= 1; i--) {
        int j = rng() % (i + 1);
        int temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

} // namespace BrainBlocks

#endif // UTILS_HPP
