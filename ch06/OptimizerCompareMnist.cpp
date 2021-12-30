#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Random.h>
#include <MultiLayerNet.h>
#include <Optimizer.h>
#include <Gnuplot.h>

static constexpr int ITERS_NUM  = 2000;
static constexpr int BATCH_SIZE = 128;

static void process(OptType type, const MatrixXd& XTrain, const VectorXi& tTrain, const MatrixXd& XTest, const VectorXi& tTest) {
    std::vector<std::unique_ptr<Optimizer>> opts;
    FILE* fp = nullptr;
    switch (type) {
    case OptType::SGD: {
        fp = std::fopen("./data/mnist_SGD.txt", "w");
        for (std::size_t i = 0; i < 5; ++i) {
            opts.push_back(std::make_unique<SGD>(0.01));
        }

        break;
    }
    case OptType::Momentum: {
        fp = std::fopen("./data/mnist_Momentum.txt", "w");
        for (std::size_t i = 0; i < 5; ++i) {
            opts.push_back(std::make_unique<Momentum>(0.01, 0.9));
        }

        break;
    }
    case OptType::AdaGrad: {
        fp = std::fopen("./data/mnist_AdaGrad.txt", "w");
        for (std::size_t i = 0; i < 5; ++i) {
            opts.push_back(std::make_unique<AdaGrad>(0.01));
        }

        break;
    }
    case OptType::Adam: {
        fp = std::fopen("./data/mnist_Adam.txt", "w");
        for (std::size_t i = 0; i < 5; ++i) {
            opts.push_back(std::make_unique<Adam>(0.001, 0.9, 0.999));
        }

        break;
    }
    default: {
        break;
    }
    }

    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    MultiLayerNet net(784, 4, 100, 10, WeightType::He, 0, 0, std::move(opts));  

    for (int i = 0; i < ITERS_NUM; ++i) {
        const std::vector<std::size_t> batchIndex = choice(XTrain.rows(), BATCH_SIZE);

        const MatrixXd XBatch = createMatrixXdBatch(XTrain, batchIndex);
        const VectorXi tBatch = createVectorXiBatch(tTrain, batchIndex);

        net.gradient(XBatch, tBatch);
        net.update();

        std::fprintf(fp, "%lf\n", net.loss(XBatch, tBatch));

        if (i % 100 == 0) {
            const double trainAcc = net.accuracy(XTrain, tTrain);
            const double testAcc = net.accuracy(XTest, tTest);
            std::printf("train acc, test acc | %lf, %lf\n", trainAcc, testAcc);
        }
    }   

    std::fclose(fp);
}

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

    srand(time(NULL));

    process(OptType::SGD, XTrain, tTrain, XTest, tTest);
    process(OptType::Momentum, XTrain, tTrain, XTest, tTest);
    process(OptType::AdaGrad, XTrain, tTrain, XTest, tTest);
    process(OptType::Adam, XTrain, tTrain, XTest, tTest);

    plotGpFile("optimization_compare_mnist.gp");

    return 0;   
}
