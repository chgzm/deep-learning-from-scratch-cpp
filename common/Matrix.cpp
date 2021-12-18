#include "Matrix.h" 

#include <fstream>
#include <random>
#include <float.h>
#include <cstdint>

int initMatrix(Eigen::MatrixXd& m, const std::string& filePath) {
    std::ifstream ifs(filePath);
    if (!ifs) {
        return -1;
    }

    double d = 0.0;
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            ifs >> d;
            m(i, j) = d;
        }
    }

    return 0;
}

int initRowVector(RowVectorXd& v, const std::string& filePath) {
    std::ifstream ifs(filePath);
    if (!ifs) {
        return -1;
    }

    double d = 0.0;
    for (int i = 0; i < v.size(); ++i) {
        ifs >> d;
        v(i) = d;
    }

    return 0;
}

MatrixXd createMatrixXdBatch(const MatrixXd& M, const std::vector<std::size_t>& index) {
    MatrixXd N(index.size(), M.cols());

    for (int i = 0; i < N.rows(); ++i) {
        for (int j = 0; j < N.cols(); ++j) {
            N(i, j) = M(index[i], j);
        }
    }

    return N;
}


VectorXi createVectorXiBatch(const VectorXi& v, const std::vector<std::size_t>& index) {
    VectorXi u(index.size());

    for (int i = 0; i < u.size(); ++i) {
        u(i) = v(index[i]);
    }

    return u;
}

int argmax(const RowVectorXd& v) {
    double max = DBL_MIN;
    int index = 0;
    for (int i = 0; i < v.size(); ++i) {
        if (max < v(i)) {
            index = i;
            max = v(i);
        }
    }
    return index;
}

RowVectorXi argmax(const MatrixXd& M) {
    RowVectorXi v(M.rows());
    for (int i = 0; i < M.rows(); ++i) {
        double max = 0.0;
        int index = 0;
        for (int j = 0; j < M.cols(); ++j) {
            if (j == 0) {
                max = M(i, j);
                continue;
            } 

            if (max < M(i, j)) {
                index = j;
                max = M(i, j);
            }
        }
        v(i) = index;
    }

    return v;
}
