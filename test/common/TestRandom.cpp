#include "gtest/gtest.h"
#include <Random.h>
#include <set>

TEST(choice, success) {
    std::vector<std::size_t> nums = choice(100, 98);

    EXPECT_EQ(98, nums.size());

    std::set<int> s;
    for (int i = 0; i < 98; ++i) {
        EXPECT_LE(0, nums[i]);
        EXPECT_LE(nums[i], 99);
        
        EXPECT_EQ(0, s.count(nums[i]));
        s.insert(nums[i]);
    }
}
