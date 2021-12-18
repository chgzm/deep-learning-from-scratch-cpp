#ifndef FUNCTION_H
#define FUNCTION_H

#include <vector>
#include <Eigen/Dense>

using RowVectorXd = Eigen::RowVectorXd;
using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

double sigmoid(double x);
RowVectorXd sigmoid(const RowVectorXd& v);
MatrixXd sigmoid(const MatrixXd& M);
MatrixXd sigmoidGrad(const MatrixXd& M);

RowVectorXd softmax(const RowVectorXd& v);
MatrixXd softmax(const MatrixXd& M);

#endif
