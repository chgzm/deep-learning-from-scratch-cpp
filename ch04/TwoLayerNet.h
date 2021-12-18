#ifndef TWOLAYERNET_H
#define TWOLAYERNET_H

#include <cstdint>
#include <Matrix.h>
#include <Eigen/Dense>

using MatrixXd    = Eigen::MatrixXd;
using MatrixXi    = Eigen::MatrixXi;
using RowVectorXd = Eigen::RowVectorXd;

class TwoLayerNet {
public:
    TwoLayerNet(std::size_t inputSize, std::size_t hiddenSize, std::size_t outputSize);
    ~TwoLayerNet() = default;

    void numericalGradient(const MatrixXd& X, const VectorXi& t); 
    void gradient(const MatrixXd& X, const VectorXi& t);
    void update(double lr);
    double accuracy(const MatrixXd& X, const VectorXi& v);

private:
    void numericalGradientMatrix(const MatrixXd& X, const VectorXi& t, MatrixXd& W, MatrixXd& dW); 
    void numericalGradientVector(const MatrixXd& X, const VectorXi& t, RowVectorXd& b, RowVectorXd& db); 

    MatrixXd predict(const MatrixXd& X);  
    double loss(const MatrixXd& X, const VectorXi& t);
    double crossEntropyError(const MatrixXd& X, const VectorXi& t);

private:
    static constexpr double WEIGHT_INIT_STD = 0.01; 

    MatrixXd    W1_;
    RowVectorXd b1_;
    MatrixXd    W2_;
    RowVectorXd b2_;

    MatrixXd    dW1_;
    RowVectorXd db1_;
    MatrixXd    dW2_;
    RowVectorXd db2_;
};

#endif
