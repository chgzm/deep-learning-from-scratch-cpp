#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Random.h>
#include <MultiLayerNet.h>
#include <Optimizer.h>
#include <Gnuplot.h>

static constexpr std::size_t ITERS_NUM  = 2000;
static constexpr std::size_t BATCH_SIZE = 128;

static void process(WeightType type, const MatrixXd& XTrain, const VectorXi& tTrain, const MatrixXd& XTest, const VectorXi& tTest) {
    std::vector<std::unique_ptr<Optimizer>> opts;
    for (std::size_t i = 0; i < 5; ++i) {
        opts.push_back(std::make_unique<SGD>(0.1));
    }

    MultiLayerNet net(784, 4, 100, 10, type, 0, 0, std::move(opts));  

    std::string filename;
    switch (type) {
    case WeightType::STD:    { filename = "./data/weight_init_compare_STD.txt";    break; }
    case WeightType::Xavier: { filename = "./data/weight_init_compare_Xavier.txt"; break; }
    case WeightType::He:     { filename = "./data/weight_init_compare_He.txt";     break; }
    default:                 { break; }
    }

    FILE* fp = std::fopen(filename.c_str(), "w");
    if (!fp) {
        std::fprintf(stderr, "Failed to open \"%s\".\n", filename.c_str());
        return;
    }

    for (std::size_t i = 0; i < ITERS_NUM; ++i) {
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
    process(WeightType::STD, XTrain, tTrain, XTest, tTest);
    process(WeightType::Xavier, XTrain, tTrain, XTest, tTest);
    process(WeightType::He, XTrain, tTrain, XTest, tTest);

    plotGpFile("plot_weight_init_compare.gp");

    return 0;   
}
