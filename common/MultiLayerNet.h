#ifndef MULTILAYERNET_H
#define MULTILAYERNET_H

#include "Net.h"
#include "Matrix.h"
#include "Layers.h"
#include "Optimizer.h"
#include "WeightType.h"
#include <memory>

class MultiLayerNet : public Net {
public:
    MultiLayerNet(
        std::size_t inputSize, 
        std::size_t hiddenLayerNum, 
        std::size_t hiddenSize, 
        std::size_t outputSize,
        WeightType weightType,
        double weight,
        double weightDecayLambda,
        std::vector<std::unique_ptr<Optimizer>>&& opt 
    );
    
    ~MultiLayerNet() {}

    void gradient(const MatrixXd& X, const VectorXi& t) override;
    void update() override;
    double accuracy(const MatrixXd& X, const VectorXi& t) override;
    double loss(const MatrixXd& X, const VectorXi& t, bool trainFlg=false) override;

private:
    MatrixXd predict(const MatrixXd& X, bool trainFlg=false);  

private:
    std::vector<Affine> A_;
    std::vector<Relu> R_;
    SoftmaxWithLoss S_;
    double weightDecayLambda_;
    std::vector<std::unique_ptr<Optimizer>> opts_;
};

#endif
