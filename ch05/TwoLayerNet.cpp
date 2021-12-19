#include "TwoLayerNet.h"
#include <Function.h>
#include <MNIST.h>

TwoLayerNet::TwoLayerNet(std::size_t inputSize, std::size_t hiddenSize, std::size_t outputSize) {
    A1_ = Affine(inputSize, hiddenSize);
    R_  = Relu();
    A2_ = Affine(hiddenSize,  outputSize);
    S_  = SoftmaxWithLoss();
}

void TwoLayerNet::gradient(const MatrixXd& X, const VectorXi& t) {
    this->loss(X, t);

    const MatrixXd X1 = S_.backward();
    const MatrixXd X2 = A2_.backward(X1);  
    const MatrixXd X3 = R_.backward(X2);  
    const MatrixXd X4 = A1_.backward(X3);  
}

void TwoLayerNet::update(double lr) {
    A1_.update(lr);
    A2_.update(lr);
}

double TwoLayerNet::accuracy(const MatrixXd& X, const VectorXi& t) {
    const MatrixXd Y = this->predict(X);
    std::size_t cnt = 0;
    const RowVectorXi y = argmax(Y);
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) == t(i)) {
            ++cnt;
        }
    }

    return (double)cnt / X.rows();
}


MatrixXd TwoLayerNet::predict(const MatrixXd& X) {
    const MatrixXd X1 = A1_.forward(X);
    const MatrixXd X2 = R_.forward(X1);
    const MatrixXd X3 = A2_.forward(X2);

    return X3;
}

double TwoLayerNet::loss(const MatrixXd& X, const VectorXi& t) {
    const MatrixXd Y = this->predict(X);
    return S_.forward(Y, t);
}
