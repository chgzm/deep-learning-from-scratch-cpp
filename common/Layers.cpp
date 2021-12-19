#include "Layers.h"
#include "Function.h"

Affine::Affine(std::size_t inputSize, std::size_t outputSize) {
    W_ = MatrixXd::Random(inputSize, outputSize);
    b_ = VectorXd::Zero(outputSize);
    dW_ = MatrixXd::Random(inputSize, outputSize);
    db_ = VectorXd::Zero(outputSize);
}

MatrixXd Affine::forward(const MatrixXd& X) {
    X_ = X;

    MatrixXd B = X * W_;
    B.rowwise() += b_;

    return B;
}

MatrixXd Affine::backward(const MatrixXd& D) {
    dW_ = X_.transpose() * D;
    db_ = D.colwise().sum();

    const MatrixXd dX = D * W_.transpose();
    return dX;
}
    
void Affine::update(double lr) {
    W_ -= dW_ * lr;
    b_ -= db_ * lr;
}

MatrixXd Relu::forward(const MatrixXd& X) {
    std::function<double(double)> f = [](double x) { return (x <= 0) ? 0.0 : 1.0;};
    M_ =  X.unaryExpr(f);
    return X.array() * M_.array();
}

MatrixXd Relu::backward(const MatrixXd& D) {
    return D.array() * M_.array();
}

SoftmaxWithLoss::SoftmaxWithLoss() {
}

double SoftmaxWithLoss::forward(const MatrixXd& X, const VectorXi& t) {
    t_ = t;
    Y_ = softmax(X);
    return this->crossEntropyError(Y_, t_); 
}

MatrixXd SoftmaxWithLoss::backward() {
    MatrixXd dX = Y_;
    for (int i = 0; i < dX.rows(); ++i) {
        for (int j = 0; j < dX.cols(); ++j) {
            if (j == t_(i)) {
                dX(i, j) -= 1.0;
            }
        }
    }

    return dX / t_.size();
}

double SoftmaxWithLoss::crossEntropyError(const MatrixXd& Y, const VectorXi& t) const {
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
