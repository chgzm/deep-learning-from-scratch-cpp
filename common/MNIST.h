#ifndef MNIST_H
#define MNIST_H

#include <cstdint>
#include <memory>
#include <Eigen/Dense>

class MNIST {
public:
    MNIST() = default;
    ~MNIST() = default;

    int load(const std::string& imagePath, const std::string& labelPath);

    inline const Eigen::MatrixXd& getImages() const {
        return images_;
    }

    inline const Eigen::VectorXi& getLabels() const {
        return labels_;
    }

private:
    int loadImage(const std::string& imagePath);
    int loadLabel(const std::string& labelPath);

private:
    constexpr static int LABEL_MAGIC = 0x00000801;
    constexpr static int IMAGE_MAGIC = 0x00000803;
    constexpr static int NUM_OF_ROWS = 28;
    constexpr static int NUM_OF_COLS = 28;
    constexpr static int NUM_OF_PIXELS = 784;

    Eigen::MatrixXd images_; 
    Eigen::VectorXi labels_; 
};

#endif

