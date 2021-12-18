#include "gtest/gtest.h"
#include <ByteReader.h>

TEST(readUInt8, success) {
    uint8_t dat[3] = {0, 1, 2};
    uint8_t* addr = (uint8_t*)(&dat);

    std::size_t pos = 0;
    EXPECT_EQ(0, readUInt8(addr, pos));
    EXPECT_EQ(1, readUInt8(addr, pos));
    EXPECT_EQ(2, readUInt8(addr, pos));
}

TEST(readInt32, success) {
    uint8_t dat[] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2};
    uint8_t* addr = (uint8_t*)(&dat);

    std::size_t pos = 0;
    EXPECT_EQ(0, readInt32(addr, pos));
    EXPECT_EQ(1, readInt32(addr, pos));
    EXPECT_EQ(2, readInt32(addr, pos));
}
