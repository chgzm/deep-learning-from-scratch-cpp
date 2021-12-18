#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Random.h>
#include <Eigen/Dense>
#include "TwoLayerNet.h"

static constexpr std::size_t ITERS_NUM  = 10000;
static constexpr std::size_t BATCH_SIZE = 100;
static constexpr double LEARNING_RATE = 0.1;

int main() {
    MNIST mnistTrain;
    if (mnistTrain.load("./../dataset/train-images-idx3-ubyte", "./../dataset/train-labels-idx1-ubyte") != 0) {
        std::fprintf(stderr, "Failed to load MNIST training set.\n");
        return -1;
    }

    MNIST mnistTest;
    if (mnistTest.load("./../dataset/t10k-images-idx3-ubyte", "./../dataset/t10k-labels-idx1-ubyte") != 0) {
        std::fprintf(stderr, "Failed to load MNIST test set.\n");
        return -1;
    }

    const MatrixXd& XTrain = mnistTrain.getImages();
    const VectorXi& tTrain = mnistTrain.getLabels();  
    const MatrixXd& XTest = mnistTest.getImages();
    const VectorXi& tTest = mnistTest.getLabels();  

    const std::size_t iterPerEpoch = std::max(XTrain.rows() / BATCH_SIZE, 1ul);

    TwoLayerNet network(784, 50, 10);
    for (std::size_t i = 0; i < ITERS_NUM; ++i) {
        const std::vector<std::size_t> batchIndex = choice(XTrain.rows(), BATCH_SIZE);

        const MatrixXd XBatch = createMatrixXdBatch(XTrain, batchIndex);
        const VectorXi tBatch = createVectorXiBatch(tTrain, batchIndex);

        network.gradient(XBatch, tBatch);
        network.update(LEARNING_RATE);

        if (i % iterPerEpoch == 0) {
            const double trainAcc = network.accuracy(XTrain, tTrain);
            const double testAcc = network.accuracy(XTest, tTest);
            std::printf("train acc, test acc | %lf, %lf\n", trainAcc, testAcc);
        }
    }   
  
    return 0;
}
