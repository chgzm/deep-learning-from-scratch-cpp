#ifndef BYTEREADER_H
#define BYTEREADER_H

#include <cstdint>

uint8_t readUInt8(const uint8_t* addr, std::size_t& pos);
int32_t readInt32(const uint8_t* addr, std::size_t& pos);

#endif
