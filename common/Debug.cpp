#include "Debug.h"

#include <cstdio>
#include <time.h>
#include <stdarg.h>

void _debug_print_tmsp() {
    struct timespec ts; 
    clock_gettime(CLOCK_REALTIME, &ts);

    struct tm tm;
    localtime_r(&(ts.tv_sec), &tm);

    char buf[20];
    const int r = std::snprintf(
        buf
      , 20
      , "%4d/%02d/%02d %02d:%02d:%02d"
      , tm.tm_year + 1900 , tm.tm_mon + 1
      , tm.tm_mday
      , tm.tm_hour
      , tm.tm_min
      , tm.tm_sec
    );
    if (r != 19) {
        std::fprintf(stderr, "snprintf failed.\n");
        return;
    }

    std::fprintf(stdout, "[%s ", buf);
}

void _debug(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
}

