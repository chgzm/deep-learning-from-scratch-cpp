#include "Random.h"
#include <random>
#include <algorithm>
#include <stdexcept>

std::vector<std::size_t> choice(std::size_t size, std::size_t num) {
    if (size < num) {
        throw std::invalid_argument("Invalid size");
    }

    std::vector<std::size_t> v(size);
    std::iota(v.begin(), v.end(), 0);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::shuffle(v.begin(), v.end(), engine);
   
    std::vector<std::size_t> ret(v.begin(), v.begin() + num);
    return ret;
}
