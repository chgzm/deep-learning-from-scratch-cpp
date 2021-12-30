#include "TwoLayerNet.h"
#include "Function.h"

#include <iostream>

TwoLayerNet::TwoLayerNet(std::size_t inputSize, std::size_t hiddenSize, std::size_t outputSize) {
    W1_ = createMatrixXdRandNormal(inputSize, hiddenSize) * WEIGHT_INIT_STD;
    b1_ = RowVectorXd::Zero(hiddenSize);
    W2_ = createMatrixXdRandNormal(hiddenSize, outputSize) * WEIGHT_INIT_STD;
    b2_ = RowVectorXd::Zero(outputSize);

    dW1_ = MatrixXd::Zero(inputSize, hiddenSize);
    db1_ = RowVectorXd::Zero(hiddenSize);
    dW2_ = MatrixXd::Zero(hiddenSize, outputSize);
    db2_ = RowVectorXd::Zero(outputSize);
}

void TwoLayerNet::numericalGradient(const MatrixXd& X, const VectorXi& t) {
    this->numericalGradientMatrix(X, t, W1_, dW1_);
    this->numericalGradientMatrix(X, t, W2_, dW2_);
    this->numericalGradientVector(X, t, b1_, db1_);
    this->numericalGradientVector(X, t, b1_, db2_);
}

void TwoLayerNet::numericalGradientMatrix(const MatrixXd& X, const VectorXi& t, MatrixXd& W, MatrixXd& dW) {
    static constexpr double h = 1e-4;
    for (int i = 0; i < W.rows(); ++i) {
        for (int j = 0; j < W.cols(); ++j) {
            const double tmp = W(i, j);
            W(i, j) = tmp + h;
            const double f1 = this->loss(X, t);

            W(i, j) = tmp - 2*h;
            const double f2 = this->loss(X, t);

            dW(i, j) = (f1 - f2) / (2*h);
            W(i, j) = tmp;
        }          
    }
}

void TwoLayerNet::numericalGradientVector(const MatrixXd& X, const VectorXi& t, RowVectorXd& b, RowVectorXd& db) {
    static constexpr double h = 1e-4;
    for (int i = 0; i < b.size(); ++i) {
        const double tmp = b(i);
        b(i) = tmp + h;
        const double f1 = this->loss(X, t);

        b(i) = tmp - 2*h;
        const double f2 = this->loss(X, t);

        db(i) = (f1 - f2) / (2*h);
        b(i) = tmp;
    }
}

void TwoLayerNet::gradient(const MatrixXd& X, const VectorXi& t) {
    // forward
    MatrixXd A1 = X * W1_;
    A1.rowwise() += b1_;

    const MatrixXd Z1 = sigmoid(A1);
    MatrixXd A2 = Z1 * W2_;
    A2.rowwise() += b2_;
    const MatrixXd Y = softmax(A2);

    // backword
    MatrixXd dY = Y;
    for (int i = 0; i < dY.rows(); ++i) {
        for (int j = 0; j < dY.cols(); ++j) {
            if (j == t(i)) {
                dY(i, j) -= 1.0;
            }
        }
    }

    dY /= X.rows();

    dW2_ = Z1.transpose() * dY;
    db2_ = dY.colwise().sum();

    const MatrixXd dZ1 = dY * W2_.transpose();
    const MatrixXd dA1 = sigmoidGrad(A1).array() * dZ1.array();
    dW1_ = X.transpose() * dA1;
    db1_ = dA1.colwise().sum(); 
}

void TwoLayerNet::update(double lr) {
    W1_ -= dW1_ * lr;   
    b1_ -= db1_ * lr;   
    W2_ -= dW2_ * lr;   
    b2_ -= db2_ * lr;   
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
    MatrixXd A1 = X * W1_;
    A1.rowwise() += b1_;

    const MatrixXd Z1 = sigmoid(A1);
    MatrixXd A2 = Z1 * W2_;
    A2.rowwise() += b2_;

    return softmax(A2);
}

double TwoLayerNet::loss(const MatrixXd& X, const VectorXi& t) {
    const MatrixXd Y = this->predict(X);
    return this->crossEntropyError(Y, t);
}

double TwoLayerNet::crossEntropyError(const MatrixXd& Y, const VectorXi& t) {
    static constexpr double delta = 1e-7;

    MatrixXd M = Y.array() + delta;
    M = M.unaryExpr([](double x) { return std::log(x);});

    double sum = 0.0;
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            if (j == t(i)) {
                sum += M(i, j);
                break;
            }
        }
    }

    return -1.0 * sum / Y.rows();
}
