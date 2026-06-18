#include <gtest/gtest.h>

// Sample test suite to verify gtest is configured correctly
TEST(SampleTest, Addition) {
    EXPECT_EQ(1 + 1, 2);
}

TEST(SampleTest, StringComparison) {
    EXPECT_STREQ("aether", "aether");
}
