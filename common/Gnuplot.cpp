#include "Gnuplot.h"
#include <cstdio>

void plotGpFile(const std::string& filePath) {
    FILE* gp = popen("gnuplot -persist", "w");
    if (gp == nullptr) {
        std::fprintf(stderr, "Failed to open gnuplot pipe.\n");
        return;
    }
    std::fprintf(gp, "load '%s'\n", filePath.c_str());
    std::fflush(gp);
    pclose(gp);
}

