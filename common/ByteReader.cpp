#include "ByteReader.h"
#include <byteswap.h>

uint8_t readUInt8(const uint8_t* addr, std::size_t& pos) {
    const uint8_t val = *(uint8_t*)(&(addr[pos]));
    ++pos;
    return val;
}

int32_t readInt32(const uint8_t* addr, std::size_t& pos) {
    const int32_t val = __bswap_32(*(int32_t*)(&(addr[pos])));
    pos += 4;
    return val;
}
