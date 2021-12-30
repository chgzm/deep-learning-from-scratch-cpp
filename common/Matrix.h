#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Dense>

using RowVectorXd = Eigen::RowVectorXd;
using RowVectorXi = Eigen::RowVectorXi;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;
using MatrixXd = Eigen::MatrixXd;

int initMatrix(MatrixXd& M, const std::string& filePath);   
int initRowVector(RowVectorXd& v, const std::string& filePath);

MatrixXd createMatrixXdRandNormal(int rows, int cols);
MatrixXd createMatrixXdRand(int rows, int cols);

MatrixXd createMatrixXdBatch(const MatrixXd& M, const std::vector<std::size_t>& index);
VectorXi createVectorXiBatch(const VectorXi& v, const std::vector<std::size_t>& index);

int argmax(const RowVectorXd& v);
RowVectorXi argmax(const MatrixXd& M);

#endif
