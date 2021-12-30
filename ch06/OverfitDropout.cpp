#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Random.h>
#include <MultiLayerNetExtend.h>
#include <Optimizer.h>
#include <Gnuplot.h>
#include <Trainer.h>
#include <Debug.h>

static constexpr std::size_t TRAIN_SIZE = 300;
static constexpr std::size_t MINI_BATCH_SIZE = 100;
static constexpr std::size_t EPOCHS = 301;
static constexpr double LEARNING_RATE = 0.01;
static constexpr double DROPOUT_RATIO = 0.4;

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

    std::vector<std::unique_ptr<Optimizer>> opts;
    for (std::size_t i = 0; i < 19; ++i) {
        opts.push_back(std::make_unique<SGD>(LEARNING_RATE));
    }

    std::unique_ptr<MultiLayerNetExtend> net = std::make_unique<MultiLayerNetExtend>(784, 6, 100, 10, WeightType::He, 0.0, true, DROPOUT_RATIO, std::move(opts));

    Trainer trainer(std::move(net), XTrain, tTrain, XTest, tTest, EPOCHS, MINI_BATCH_SIZE, true);
    trainer.train();

    FILE* fp = fopen("./data/overfit_dropout.txt", "w");
    if (!fp) {
        std::fprintf(stderr, "failed to open file.\n");
        return -1;
    }

    for (std::size_t i = 0; i < EPOCHS; ++i) {
        std::fprintf(fp, "%lu %lf %lf\n", i, trainer.getTrainAccList()[i], trainer.getTestAccList()[i]);
    }

    std::fclose(fp);

    plotGpFile("plot_overfit_dropout.gp");

    return 0;   
}
