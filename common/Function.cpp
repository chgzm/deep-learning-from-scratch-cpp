#include "Function.h"

#include <cmath>
#include <stdexcept>
#include <functional>
#include <iostream>


static double _sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

RowVectorXd sigmoid(const RowVectorXd& v) {
    RowVectorXd u(v.size());

    for (int i = 0; i < v.size(); ++i) {
        u(i) = sigmoid(v(i));
    }

    return u;
}

MatrixXd sigmoid(const MatrixXd& M) {
    std::function<double(double)> func = _sigmoid;
    MatrixXd N = M.unaryExpr(func);

    return N;
}

MatrixXd sigmoidGrad(const MatrixXd& M) {
    std::function<double(double)> func = _sigmoid;
    MatrixXd N = M.unaryExpr(func);
    MatrixXd Z = MatrixXd::Zero(M.rows(), M.cols());
    std::function<double(double)> f = [](double x) { return x + 1;};
    MatrixXd O = Z.unaryExpr(f);

    return (O - N).array() * N.array();
}

RowVectorXd softmax(const RowVectorXd& v) {
    double max = 0.0;
    for (int i = 0; i < v.size(); ++i) {
        max = std::max(max, v(i));
    }

    double sum = 0.0;
    for (int i = 0; i < v.size(); ++i) {
        sum += std::exp(v(i) - max);
    }

    RowVectorXd u(v.size());
    for (int i = 0; i < v.size(); ++i) {
        const double val = std::exp(v(i) - max) / sum;
        u(i) = val;
    }

    return u;
}

double func(double val, double max, double sum) {
    return std::exp(val - max) / sum;
}

MatrixXd softmax(const MatrixXd& M) {
    MatrixXd N = M;
    VectorXd maxs = M.rowwise().maxCoeff();
    N.colwise() -= maxs;

    std::function<double(double)> f = [](double x) { return std::exp(x); };
    N = N.unaryExpr(f);
    VectorXd sums = N.rowwise().sum();

    // N = N.colwise() /= sums;

    for (int i = 0; i < N.rows(); ++i) {
        for (int j = 0; j < N.cols(); ++j) {
            N(i, j) /= sums(i);
        }
    }
   
    return N; 
}
