#include "Util.h"
#include "gtest/gtest.h"

void EXPECT_MATRIX_NEAR(const std::vector<std::vector<double>>& E, const MatrixXd& M) {
    EXPECT_EQ(E.size(), M.rows());
    EXPECT_EQ(E[0].size(), M.cols());

    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            EXPECT_NEAR(E[i][j], M(i, j), 10e-8);
        }
    }
}
