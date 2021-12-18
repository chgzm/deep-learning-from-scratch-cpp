#include "gtest/gtest.h"
#include <MNIST.h>

TEST(load_mnist_images, success) {
    MNIST mnist;
    EXPECT_EQ(0, mnist.load("../../dataset/t10k-images-idx3-ubyte", "../../dataset/t10k-labels-idx1-ubyte"));
    EXPECT_EQ(10000, mnist.getImages().rows());
    EXPECT_EQ(784, mnist.getImages().cols());
    EXPECT_EQ(10000, mnist.getLabels().size());
}

TEST(load_mnist_images, error) {
    MNIST mnist;
    EXPECT_EQ(-1, mnist.load("foo.dat", "bar.dat"));
    EXPECT_EQ(-1, mnist.load("../../dataset/t10k-images-idx3-ubyte", "bar.dat"));
    EXPECT_EQ(-1, mnist.load("foo.dat", "../../dataset/t10k-labels-idx1-ubyte"));
}
