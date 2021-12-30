#ifndef LAYERS_H
#define LAYERS_H

#include "Matrix.h"
#include "Optimizer.h"

class Affine {
public:
    Affine(std::size_t inputSize, std::size_t outputSize);
    Affine() = default;
    ~Affine() = default;

    MatrixXd forward(const MatrixXd& X);
    MatrixXd backward(const MatrixXd& D);
    void update(double lr);

public:
    MatrixXd W_;  
    MatrixXd dW_;  
    RowVectorXd b_;
    RowVectorXd db_;

private:
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
    SoftmaxWithLoss() = default;
    ~SoftmaxWithLoss() = default;

    double forward(const MatrixXd& X, const VectorXi& t);
    MatrixXd backward();

private:
    double crossEntropyError(const MatrixXd& Y, const VectorXi& t) const;

private:
    MatrixXd Y_;
    VectorXi t_;
};

class BatchNormalization {
public:
    BatchNormalization(std::size_t size1, std::size_t size2, double momentum);
    ~BatchNormalization() = default;

    MatrixXd forward(const MatrixXd& X);
    MatrixXd backward(const MatrixXd& D);

public:
    RowVectorXd g_;
    RowVectorXd b_;
    RowVectorXd dg_;
    RowVectorXd db_;  

private:
    MatrixXd xc_;
    MatrixXd xn_;
    RowVectorXd std_;
    RowVectorXd runningMean_;
    RowVectorXd runningVar_;

    double momentum_;
    int batchSize_ = 0;
    bool isInit_ = false;
};

class Dropout {
public:
    Dropout(double ratio);
    ~Dropout() = default;

    MatrixXd forward(const MatrixXd& X, bool trainFlag);
    MatrixXd backward(const MatrixXd& D);

private:
    MatrixXd M_;
    double ratio_;
};

#endif
