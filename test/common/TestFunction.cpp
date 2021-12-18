#include "gtest/gtest.h"
#include <Function.h>

TEST(vector_sigmoid, success) {
    RowVectorXd v(3);
    v << -1 , 1 , 2;

    const RowVectorXd u = sigmoid(v);

    EXPECT_DOUBLE_EQ(sigmoid(-1), u(0));
    EXPECT_DOUBLE_EQ(sigmoid(1), u(1));
    EXPECT_DOUBLE_EQ(sigmoid(2), u(2));
}

TEST(matrix_sigmoid, success) {
    MatrixXd M(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            M(i, j) = i * 3 + j - 2;
        }
    }

    const MatrixXd N = sigmoid(M);

    for (int i = 0; i < N.rows(); ++i) {
        for (int j = 0; j < N.cols(); ++j) {
            EXPECT_DOUBLE_EQ(sigmoid(M(i, j)), N(i, j));
        }
     }
}

TEST(sigmoid_grad, success) {
    MatrixXd M(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            M(i, j) = i * 3 + j - 2;
        }
    }

    const MatrixXd N = sigmoidGrad(M);
 
    double ans[2][3] = {{0.10499359, 0.19661193, 0.25}, {0.19661193, 0.10499359, 0.04517666}};

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(ans[i][j], N(i, j), 10e-8);
        }
    }
}

TEST(vector_softmax, success) {
    RowVectorXd v(3);
    v << 1 , 2 , 3;

    const RowVectorXd u = softmax(v);
    EXPECT_NEAR(0.09003057, u(0), 10e-8);
    EXPECT_NEAR(0.24472847, u(1), 10e-8);
    EXPECT_NEAR(0.66524096, u(2), 10e-8);
}

TEST(matrix_softmax, success) {
    MatrixXd M(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            M(i, j) = i * 3 + j - 2;
        }
    }

    const MatrixXd N = softmax(M);
    double ans[2][3] = {{0.09003057, 0.24472847, 0.66524096}, {0.09003057, 0.24472847, 0.66524096}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(ans[i][j], N(i, j), 10e-8);
        }
    }
}
