#ifndef DEBUG_H
#define DEBUG_H

#define debug(fmt, ...) \
    _debug_print_tmsp(); \
    std::fprintf(stdout, "%s:%d %s] ", __FILE__, __LINE__, __func__); \
    _debug(fmt, ## __VA_ARGS__); \
    std::fflush(stdout)

void _debug_print_tmsp();
void _debug(const char* fmt, ...);

#endif
