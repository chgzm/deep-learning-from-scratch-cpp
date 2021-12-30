#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Random.h>
#include <MultiLayerNet.h>
#include <Optimizer.h>
#include <Gnuplot.h>
#include <Trainer.h>
#include <Debug.h>

static constexpr std::size_t OPTIMIZATION_TRIAL = 100;
static constexpr std::size_t TRAIN_SIZE = 400;
static constexpr std::size_t VALIDATION_NUM = 100;
static constexpr std::size_t EPOCHS = 50;
static constexpr std::size_t MINI_BATCH_SIZE = 100;

static void process(
    const MatrixXd& XTrain,
    const VectorXi& tTrain,
    const MatrixXd& XVal,
    const VectorXi& tVal,
    double lr,
    double decay,
    std::size_t trial,
    double trainResult[OPTIMIZATION_TRIAL][EPOCHS],
    double valResult[OPTIMIZATION_TRIAL][EPOCHS]
) {
   
    std::vector<std::unique_ptr<Optimizer>> opts;
    for (std::size_t i = 0; i < 7; ++i) {
        opts.push_back(std::make_unique<SGD>(lr));
    }

    auto net = std::make_unique<MultiLayerNet>(784, 6, 100, 10, WeightType::He, 0.0, decay, std::move(opts));
    Trainer trainer(std::move(net), XTrain, tTrain, XVal, tVal, EPOCHS, MINI_BATCH_SIZE, false);
    trainer.train();

    for (std::size_t i = 0; i < EPOCHS; ++i) {
        trainResult[trial][i] = trainer.getTrainAccList()[i]; 
        valResult[trial][i] = trainer.getTestAccList()[i];
    }
}

struct Result {
    std::size_t index;
    double acc;
    double lr;
    double decay;
};

int main() {
    MNIST mnistTrain;
    if (mnistTrain.load("./../dataset/train-images-idx3-ubyte", "./../dataset/train-labels-idx1-ubyte") != 0) {
        std::fprintf(stderr, "Failed to load MNIST training set.\n");
        return -1;
    }

    const MatrixXd& XTrain = mnistTrain.getImages().topRows(TRAIN_SIZE);
    const VectorXi& tTrain = mnistTrain.getLabels().topRows(TRAIN_SIZE);
    const MatrixXd& XVal = mnistTrain.getImages().topRows(TRAIN_SIZE + VALIDATION_NUM).bottomRows(VALIDATION_NUM);
    const VectorXi& tVal = mnistTrain.getLabels().topRows(TRAIN_SIZE + VALIDATION_NUM).bottomRows(VALIDATION_NUM);

    std::vector<Result> results(OPTIMIZATION_TRIAL);
    double trainResult[OPTIMIZATION_TRIAL][EPOCHS];
    double valResult[OPTIMIZATION_TRIAL][EPOCHS];

    srand(time(NULL));
    for (std::size_t i = 0; i < OPTIMIZATION_TRIAL; ++i) {
        const double lr = std::pow(10, uniform(-6, -2));
        const double decay = std::pow(10, uniform(-8, -4));

        process(XTrain, tTrain, XVal, tVal, lr, decay, i, trainResult, valResult);
        results[i] = {i, valResult[i][EPOCHS-1], lr, decay};

        std::printf("val acc:  %.4lf | lr: %.8lf, weight decay: %.8lf\n", valResult[i][EPOCHS-1], lr, decay);
    }

    std::sort(results.begin(), results.end(), [](const auto& l, const auto& r) { return l.acc > r.acc; });

    printf("=========== Hyper-Parameter Optimization Result ===========\n");
    for (std::size_t i = 0; i < 20; ++i) {
        std::printf(
            "Best-%02lu(val acc:%.4lf) | lr:%.8lf, weight decay:%.8lf\n", 
            i + 1, 
            results[i].acc, 
            results[i].lr, 
            results[i].decay
        );
    }

    return 0;   
}
