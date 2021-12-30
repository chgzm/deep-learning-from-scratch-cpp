#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Random.h>
#include <MultiLayerNet.h>
#include <Optimizer.h>
#include <Gnuplot.h>

static constexpr std::size_t TRAIN_SIZE = 300;
static constexpr std::size_t BATCH_SIZE = 100;
static constexpr std::size_t MAX_EPOCHS = 201;
static constexpr double WEIGHT_DECAY_LAMBDA = 0.1;

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

    const MatrixXd& XTrain = mnistTrain.getImages().topRows(TRAIN_SIZE);
    const VectorXi& tTrain = mnistTrain.getLabels().topRows(TRAIN_SIZE);
    const MatrixXd& XTest = mnistTest.getImages();
    const VectorXi& tTest = mnistTest.getLabels();  

    srand(time(NULL));

    FILE* fp = fopen("./data/overfit_weight_decay.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return -1;
    }

    std::vector<std::unique_ptr<Optimizer>> opts;
    for (std::size_t i = 0; i < 7; ++i) {
        opts.push_back(std::make_unique<SGD>(0.01));
    }
 
    MultiLayerNet net(784, 6, 100, 10, WeightType::He, 0, WEIGHT_DECAY_LAMBDA, std::move(opts));  
    
    const std::size_t iterPerEpoch = std::max(TRAIN_SIZE / BATCH_SIZE, 1ul);
    std::size_t epochCnt = 0;
    for (int i = 0; i < 1000000000; ++i) {
        const std::vector<std::size_t> batchIndex = choice(TRAIN_SIZE, BATCH_SIZE);

        const MatrixXd XBatch = createMatrixXdBatch(XTrain, batchIndex);
        const VectorXi tBatch = createVectorXiBatch(tTrain, batchIndex);

        net.gradient(XBatch, tBatch);
        net.update();

        if (i % iterPerEpoch == 0) {
            const double trainAcc = net.accuracy(XTrain, tTrain);
            const double testAcc = net.accuracy(XTest, tTest);
            std::printf("epoch: %lu, train acc: %lf, test acc: %lf\n", epochCnt, trainAcc, testAcc);
            std::fprintf(fp, "%lu %lf %lf\n", epochCnt, trainAcc, testAcc);
            ++epochCnt;
        }

        if (epochCnt >= MAX_EPOCHS) {
            break;
        }   
    }

    std::fclose(fp);

    plotGpFile("plot_overfit_weight_decay.gp");

    return 0;   
}
