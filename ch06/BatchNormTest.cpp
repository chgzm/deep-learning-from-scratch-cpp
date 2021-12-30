#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Random.h>
#include <MultiLayerNet.h>
#include <MultiLayerNetExtend.h>
#include <Optimizer.h>
#include <Gnuplot.h>
#include <Util.h>
#include "Debug.h"
#include <iostream>

static constexpr std::size_t TRAIN_SIZE = 1000;
static constexpr std::size_t BATCH_SIZE = 100;
static constexpr std::size_t MAX_EPOCHS = 20;

static void process(const MatrixXd& XTrain, const VectorXi& tTrain) {
    const std::vector<double> weightScaleList = logspace(0, -4, 16);

    for (int i = 0; i < 16; ++i) {
        std::printf("============== %d/16 ==============\n", i + 1);

        std::vector<std::unique_ptr<Optimizer>> opts;
        for (std::size_t i = 0; i < 6; ++i) {
            opts.push_back(std::make_unique<SGD>(0.01));
        }

        std::vector<std::unique_ptr<Optimizer>> opts2;
        for (std::size_t i = 0; i < 16; ++i) {
            opts2.push_back(std::make_unique<SGD>(0.01));
        }

        char file_name[64];
        std::snprintf(file_name, 64, "./data/batch_norm_test_%.4lf.txt", weightScaleList[i]);
        FILE* fp = fopen(file_name, "w");
        if (!fp) {
            fprintf(stderr, "Failed to open file=%s\n", file_name);
            return;
        }

        MultiLayerNet net(784, 5, 100, 10, WeightType::STD, weightScaleList[i], 0, std::move(opts));  
        MultiLayerNetExtend netBN(784, 5, 100, 10, WeightType::STD, weightScaleList[i], false, 0, std::move(opts2));  

        std::size_t epochCnt = 0;
        for (int j = 0; j < 1000000000; ++j) {
            const std::vector<std::size_t> batchIndex = choice(TRAIN_SIZE, BATCH_SIZE);
            const MatrixXd XBatch = createMatrixXdBatch(XTrain, batchIndex);
            const VectorXi tBatch = createVectorXiBatch(tTrain, batchIndex);

            net.gradient(XBatch, tBatch);
            netBN.gradient(XBatch, tBatch);
            net.update();
            netBN.update();

            if (j % 10 == 0) {
                const double trainAcc = net.accuracy(XTrain, tTrain);
                const double trainAccBN = netBN.accuracy(XTrain, tTrain);

                std::printf("epoch:%lu | %lf-%lf\n", epochCnt, trainAcc, trainAccBN);
                std::fprintf(fp, "%lu %lf %lf\n", epochCnt, trainAcc, trainAccBN);
                ++epochCnt;
           }

            if (epochCnt >= MAX_EPOCHS) {
                break;
            }
        }   
        std::fclose(fp);
    }
}

int main() {
    MNIST mnistTrain;
    if (mnistTrain.load("./../dataset/train-images-idx3-ubyte", "./../dataset/train-labels-idx1-ubyte") != 0) {
        std::fprintf(stderr, "Failed to load MNIST training set.\n");
        return -1;
    }

    const MatrixXd& XTrain = mnistTrain.getImages().topRows(TRAIN_SIZE);
    const VectorXi& tTrain = mnistTrain.getLabels().topRows(TRAIN_SIZE);

    srand(time(NULL));
    process(XTrain, tTrain);

    plotGpFile("plot_batch_norm_test.gp");

    return 0;   
}
