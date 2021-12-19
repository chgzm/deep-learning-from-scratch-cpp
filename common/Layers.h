#ifndef LAYERS_H
#define LAYERS_H

#include "Matrix.h"

class Affine {
public:
    Affine(std::size_t inputSize, std::size_t outputSize);
    Affine() = default;
    ~Affine() = default;

    MatrixXd forward(const MatrixXd& X);
    MatrixXd backward(const MatrixXd& D);
    
    void update(double lr);

private:
    MatrixXd W_;  
    MatrixXd dW_;  
    RowVectorXd b_;
    RowVectorXd db_;
    MatrixXd X_;
};

class Relu {
public:
    Relu() = default;
    ~Relu() = default;

    MatrixXd forward(const MatrixXd& X);
    MatrixXd backward(const MatrixXd& D);

private:
    MatrixXd M_;
};

class SoftmaxWithLoss {
public:
    SoftmaxWithLoss();
    ~SoftmaxWithLoss() = default;

    double forward(const MatrixXd& X, const VectorXi& t);
    MatrixXd backward();

private:
    double crossEntropyError(const MatrixXd& Y, const VectorXi& t) const;

private:
    MatrixXd Y_;
    VectorXi t_;
};

#endif
