#include "gtest/gtest.h"
#include <Layers.h>
#include "Util.h"

TEST(Afine_forward_backward, success) {
    Affine A(4, 3);

    MatrixXd X(3, 4);
    X << 0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 0.0;

    const MatrixXd M = A.forward(X);
    EXPECT_MATRIX_NEAR({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, M);
    
    MatrixXd B(3, 3);
    B << 0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0, 
         0.0, 0.0, 0.0;

    const MatrixXd N = A.backward(B);
    EXPECT_MATRIX_NEAR({{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0,  0.0, 0.0, 0.0}}, N);
}

TEST(Relu_forward_backward, success) {
    Relu R;

    MatrixXd M(2, 3);
    M << -1,  1, -1,
          1, -1,  1;
    const MatrixXd A = R.forward(M);

    MatrixXd ANS(2, 3);
    ANS << 0, 1, 0, 
           1, 0, 1; 
    
    EXPECT_EQ(true, ANS == A);

    MatrixXd N(2, 3);
    N << 1, 1, 1, 
         2, 2, 2;

    const MatrixXd B = R.backward(N);
    
    MatrixXd ANS2(2, 3);
    ANS2 << 0, 1, 0,
            2, 0, 2;

    EXPECT_EQ(true, ANS2 == B);
}

TEST(Softmax_with_loss_forward_backward, success) {
    SoftmaxWithLoss S;
    
    MatrixXd X(2, 3);
    X << 1, 2, 2,
         4, 5, 6;

    VectorXi t(2);
    t << 0, 1;

    const double loss = S.forward(X, t);
    EXPECT_DOUBLE_EQ(loss, 1.6347998581152146);

    const MatrixXd M = S.backward();
    EXPECT_MATRIX_NEAR({{-0.4223188, 0.2111594, 0.2111594}, {0.04501529, -0.37763576, 0.33262048}}, M);
}

