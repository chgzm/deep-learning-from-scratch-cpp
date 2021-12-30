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

double uniform(double start, double stop) {
    double v = ((double)(rand()) + 1.0) / ((double)(RAND_MAX) + 2.0);
    v *= (stop - start);
    return v + start;
}


