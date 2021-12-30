#ifndef TRAINER_H
#define TRAINER_H

#include "Net.h"
#include <memory>

class Trainer {
public:
    Trainer(
        std::unique_ptr<Net> net, 
        const MatrixXd& XTrain, 
        const VectorXi& tTrain,
        const MatrixXd& XTest, 
        const VectorXi& tTest,
        std::size_t epochs,
        std::size_t batchSize,
        bool verbose=true
    );
    ~Trainer() = default;

    void train();

    inline const std::vector<double>& getTrainAccList() const {
        return trainAccList_;
    }

    inline const std::vector<double>& getTestAccList() const {
        return testAccList_;
    }

private:
    void step(std::size_t iter);

private:
    std::unique_ptr<Net> net_;
    MatrixXd XTrain_;
    VectorXi tTrain_;
    MatrixXd XTest_;
    VectorXi tTest_;
    std::size_t epochs_;
    std::size_t batchSize_;
    std::vector<double> trainAccList_;
    std::vector<double> testAccList_;
    bool verbose_;
};

#endif
