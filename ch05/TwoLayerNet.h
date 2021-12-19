#ifndef TWOLAYERNET_H
#define TWOLAYERNET_H

#include <Matrix.h>
#include <Layers.h>

class TwoLayerNet {
public:
    TwoLayerNet(std::size_t inputSize, std::size_t hiddenSize, std::size_t outputSize);
    ~TwoLayerNet() = default;
    
    void gradient(const MatrixXd& X, const VectorXi& t);
    void update(double lr);
    double accuracy(const MatrixXd& X, const VectorXi& t);

private:
    MatrixXd predict(const MatrixXd& X);  
    double loss(const MatrixXd& X, const VectorXi& t);

private:
    static constexpr double WEIGHT_INIT_STD = 0.01; 

    Affine A1_;
    Relu R_;
    Affine A2_;
    SoftmaxWithLoss S_;
};

#endif
