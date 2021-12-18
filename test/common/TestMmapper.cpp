#include "gtest/gtest.h"
#include <Mmapper.h>

TEST(mmapReadOnly, success) {
    Mmapper mapper;
    uint8_t* addr = (uint8_t*)(mapper.mmapReadOnly("./data/test_file.txt"));
    EXPECT_NE(nullptr, addr);

    EXPECT_EQ('T', addr[0]);
    EXPECT_EQ('E', addr[1]);
    EXPECT_EQ('S', addr[2]);
    EXPECT_EQ('T', addr[3]);
}

TEST(mmapReadOnly, error) {
    Mmapper mapper;
    void* addr = mapper.mmapReadOnly("foo.txt");
    EXPECT_EQ(nullptr, addr);
}

