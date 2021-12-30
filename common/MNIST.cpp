#include "MNIST.h"
#include "Mmapper.h"
#include "ByteReader.h"

int MNIST::loadImage(const std::string& imagePath) {
    Mmapper mapper;
    uint8_t* addr = (uint8_t*)(mapper.mmapReadOnly(imagePath));
    if (addr == nullptr) {
        return -1;
    }

    std::size_t pos = 0;
    const int32_t magic = readInt32(addr, pos);
    if (magic != MNIST::IMAGE_MAGIC) {
        return -1; 
    }   

    const int32_t numImages = readInt32(addr, pos);

    const int32_t numRows = readInt32(addr, pos);
    if (numRows != MNIST::NUM_OF_ROWS) {
        return -1; 
    }   

    const int32_t numCols = readInt32(addr, pos);
    if (numCols != MNIST::NUM_OF_COLS) {
        return -1; 
    }   

    images_ = Eigen::MatrixXd(numImages, MNIST::NUM_OF_PIXELS);
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < MNIST::NUM_OF_PIXELS; ++j) {
            const uint8_t d = readUInt8(addr, pos);
            images_(i, j) = d / 255.0;
        }
    }

    return 0;
}

int MNIST::loadLabel(const std::string& labelPath) {
    Mmapper mapper;
    uint8_t* addr = (uint8_t*)(mapper.mmapReadOnly(labelPath));
    if (addr == nullptr) {
        return -1;
    }

    std::size_t pos = 0;
    const int32_t magic = readInt32(addr, pos);
    if (magic != MNIST::LABEL_MAGIC) {
        return -1; 
    }   

    const int32_t numLabels = readInt32(addr, pos);

    labels_ = Eigen::VectorXi(numLabels);
    for (int i = 0; i < numLabels; ++i) {
        const uint8_t l = readUInt8(addr, pos);
        labels_(i) = l;
    }

    return 0;
}

int MNIST::load(const std::string& imagePath, const std::string& labelPath) {
    if (this->loadImage(imagePath) != 0) {
        return -1;
    }

    if (this->loadLabel(labelPath) != 0) {
        return -1;
    }

    if (images_.rows() != labels_.size()) {
        return -1;
    }

    return 0;
}
