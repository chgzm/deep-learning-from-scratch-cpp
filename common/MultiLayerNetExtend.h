#ifndef MULTILAYERNETEXTEND_H
#define MULTILAYERNETEXTEND_H

#include "Net.h"
#include "Matrix.h"
#include "Layers.h"
#include "Optimizer.h"
#include "WeightType.h"
#include <memory>

class MultiLayerNetExtend : public Net {
public:
    MultiLayerNetExtend(
        std::size_t inputSize, 
        std::size_t hiddenLayerNum, 
        std::size_t hiddenSize, 
        std::size_t outputSize,
        WeightType weightType,
        double weight,
        bool useDropout,
        double dropoutRatio,    
        std::vector<std::unique_ptr<Optimizer>>&& opt 
    );
    
    ~MultiLayerNetExtend() {}

    void gradient(const MatrixXd& X, const VectorXi& t) override;
    void update() override;
    double accuracy(const MatrixXd& X, const VectorXi& t) override;
    double loss(const MatrixXd& X, const VectorXi& t, bool trainFlag=false) override;

private:
    MatrixXd predict(const MatrixXd& X, bool trainFlag=false);  

private:
    std::vector<Affine> A_;
    std::vector<Relu> R_;
    std::vector<BatchNormalization> B_;
    std::vector<Dropout> D_;
    SoftmaxWithLoss S_;
    std::vector<std::unique_ptr<Optimizer>> opts_;
    bool useDropout_;
};

#endif
