#include "Util.h"
#include <cmath>

std::vector<double> logspace(double start, double stop, int num) {
    std::vector<double> ret(num);
    const double delta = (stop - start) / (num - 1);
    double cur = start;
    for (int i = 0; i < num; ++i) {
        ret[i] = std::pow(10, cur);
        cur += delta;
    }

    return ret;
}

