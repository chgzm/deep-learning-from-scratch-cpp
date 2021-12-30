#include "gtest/gtest.h"
#include <Layers.h>
#include <Matrix.h>
#include "Util.h"

TEST(Afine_forward_backward, success) {
    Affine A(4, 3);

    A.W_ = MatrixXd(4, 3);
    A.W_ << 0.1, 0.2, 0.3, 
            0.4, 0.5, 0.6, 
            0.7, 0.8, 0.9, 
            1.0, 1.1, 1.2;

    A.b_ = RowVectorXd(3);
    A.b_ << 1, 2, 3;

    MatrixXd X(3, 4);
    X << 0.1, 0.2, 0.3, 0.4, 
         0.5, 0.6, 0.7, 0.8, 
         0.9, 1.0, 1.1, 1.2;

    const MatrixXd M = A.forward(X);
    EXPECT_MATRIX_NEAR({{1.7, 2.8, 3.9}, {2.58, 3.84, 5.1}, {3.46, 4.88, 6.3}}, M);
    
    MatrixXd B(3, 3);
    B << 0.1, 0.2, 0.3, 
         0.4, 0.5, 0.6, 
         0.7, 0.8, 0.9;

    const MatrixXd N = A.backward(B);
    EXPECT_MATRIX_NEAR({{0.14, 0.32, 0.5, 0.68}, {0.32, 0.77, 1.22, 1.67}, {0.5,  1.22, 1.94, 2.66}}, N);
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

TEST(BatchNormalization_forward_backward, success) {
    BatchNormalization B(4, 4, 0.9);
   
    MatrixXd X = MatrixXd(3, 4);
    X <<  1, 3,  2,  4, 
          8, 6,  7,  5, 
         10, 9, 11, 12;

    const MatrixXd M = B.forward(X);

    EXPECT_MATRIX_NEAR({{-1.38218943, -1.22474477, -1.2675004, -0.8429272}, {0.4319342, 0, 0.09053574, -0.56195146}, {0.95025524, 1.22474477, 1.17696466, 1.40487866}}, M);

    X *= 10000000;
    const MatrixXd N = B.backward(X);
    EXPECT_MATRIX_NEAR({{-0.92833612, -2.04124094, -0.93504121, -0.66546879}, {0.29010504, 0, 0.06678866, -0.44364586}, {0.63823109, 2.04124094, 0.86825255, 1.10911464}}, N);
}

TEST(dropout_forward_backward, success) {
    Dropout D(0.5);
    
    MatrixXd X(3, 4);
    X <<  1, 3,  2,  4, 
          8, 6,  7,  5, 
         10, 9, 11, 12;

    const MatrixXd A = D.forward(X, true);
    std::cout << A << std::endl;
    const MatrixXd B = D.forward(X, false);
    std::cout << B << std::endl;
    const MatrixXd N = D.backward(X);
    std::cout << N << std::endl;
}
