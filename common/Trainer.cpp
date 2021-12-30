#include "Trainer.h"
#include "Random.h"
#include "Debug.h"

Trainer::Trainer(
    std::unique_ptr<Net> net, 
    const MatrixXd& XTrain, 
    const VectorXi& tTrain,
    const MatrixXd& XTest, 
    const VectorXi& tTest,
    std::size_t epochs,
    std::size_t batchSize,
    bool verbose
) : net_(std::move(net)),  
    XTrain_(XTrain),
    tTrain_(tTrain),
    XTest_(XTest),
    tTest_(tTest),
    epochs_(epochs),
    batchSize_(batchSize),
    verbose_(verbose)
{
}

void Trainer::train() {
    const std::size_t iterPerEpoch = XTrain_.rows() / batchSize_;
    const std::size_t maxIter = epochs_ * iterPerEpoch;

    for (std::size_t i = 0; i < maxIter; ++i) {
        this->step(i);
    }

    const double testAcc = net_->accuracy(XTest_, tTest_);
    if (verbose_) {
        std::printf("=============== Final Test Accuracy ===============\n");
        std::printf("test acc:%lf\n", testAcc);
    }
}

void Trainer::step(std::size_t iter) {
    const std::vector<std::size_t> batchIndex = choice(XTrain_.rows(), batchSize_);
    
    const MatrixXd XBatch = createMatrixXdBatch(XTrain_, batchIndex);
    const VectorXi tBatch = createVectorXiBatch(tTrain_, batchIndex);

    net_->gradient(XBatch, tBatch);
    net_->update();

    static const std::size_t iterPerEpoch = XTrain_.rows() / batchSize_;
    static std::size_t epoch = 0;
    if (iter % iterPerEpoch == 0) {
        const double trainAcc = net_->accuracy(XTrain_, tTrain_);
        const double testAcc = net_->accuracy(XTest_, tTest_);

        if (verbose_) {
            std::printf("epoch: %lu, train acc: %lf, test acc: %lf\n", epoch, trainAcc, testAcc);
        }

        trainAccList_.push_back(trainAcc);
        testAccList_.push_back(testAcc);

        ++epoch;
    }
}

