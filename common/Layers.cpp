#include "Layers.h"
#include "Matrix.h"
#include "Function.h"
#include "Debug.h"
#include <iostream>

Affine::Affine(std::size_t inputSize, std::size_t outputSize) {
    W_ = createMatrixXdRandNormal(inputSize, outputSize);
    b_ = VectorXd::Zero(outputSize);
    dW_ = createMatrixXdRandNormal(inputSize, outputSize);
    db_ = VectorXd::Zero(outputSize);
}

MatrixXd Affine::forward(const MatrixXd& X) {
    X_ = X;

    MatrixXd B = X * W_;
    B.rowwise() += b_;

    return B;
}

MatrixXd Affine::backward(const MatrixXd& D) {
    dW_ = X_.transpose() * D;
    db_ = D.colwise().sum();

    const MatrixXd dX = D * W_.transpose();
    return dX;
}
    
void Affine::update(double lr) {
    W_ -= dW_ * lr;
    b_ -= db_ * lr;
}

MatrixXd Relu::forward(const MatrixXd& X) {
    std::function<double(double)> f = [](double x) { return (x <= 0) ? 0.0 : 1.0;};
    M_ =  X.unaryExpr(f);
    return X.array() * M_.array();
}

MatrixXd Relu::backward(const MatrixXd& D) {
    return D.array() * M_.array();
}

double SoftmaxWithLoss::forward(const MatrixXd& X, const VectorXi& t) {
    t_ = t;
    Y_ = softmax(X);
    return this->crossEntropyError(Y_, t_); 
}

MatrixXd SoftmaxWithLoss::backward() {
    MatrixXd dX = Y_;
    for (int i = 0; i < dX.rows(); ++i) {
        for (int j = 0; j < dX.cols(); ++j) {
            if (j == t_(i)) {
                dX(i, j) -= 1.0;
            }
        }
    }

    return dX / t_.size();
}

double SoftmaxWithLoss::crossEntropyError(const MatrixXd& Y, const VectorXi& t) const {
    static constexpr double delta = 1e-7;

    MatrixXd M = Y.array() + delta;
    M = M.unaryExpr([](double x) { return std::log(x);});

    double sum = 0.0;
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            if (j == t(i)) {
                sum += M(i, j);
                break;
            }
        }
    }
    
    return -1.0 * sum / Y.rows();
}

BatchNormalization::BatchNormalization(std::size_t size1, std::size_t size2, double momentum)
  : momentum_(momentum)
{
    g_ = RowVectorXd::Ones(size1);
    b_ = RowVectorXd::Zero(size2);
}

MatrixXd BatchNormalization::forward(const MatrixXd& X) {
    if (!isInit_) {
        runningMean_ = RowVectorXd::Zero(g_.size());
        runningVar_ = RowVectorXd::Zero(g_.size());
        isInit_ = true;
    }

    batchSize_ = X.rows();

    // mu
    RowVectorXd mu = X.colwise().mean();

    // xc
    xc_ = X;
    for (int i = 0; i < xc_.cols(); ++i) {
        for (int j = 0; j < xc_.rows(); ++j) {
            xc_(j, i) -= mu(i);
        }
    }

    // var 
    MatrixXd xcTmp = xc_.unaryExpr([](double x) { return std::pow(x, 2); });
    RowVectorXd var = xcTmp.colwise().mean();

    // std 
    RowVectorXd varTmp = var.unaryExpr([](double x) { return  x + 10e-7; });
    std_ = varTmp.unaryExpr([](double x) { return std::sqrt(x); });

    // xn 
    xn_ = xc_;
    for (int i = 0; i < xn_.cols(); ++i) {
        for (int j = 0; j < xn_.rows(); ++j) {
            xn_(j, i) /= std_(i);
        }
    }

    RowVectorXd prevRunningMean = runningMean_;
    RowVectorXd prevRunningVar = runningVar_;
 
    // running_mean
    mu *= (1.0 - momentum_);
    runningMean_ =  prevRunningMean + mu; 

    // running_var
    prevRunningVar *= momentum_;
    var *= (1.0 - momentum_);
    runningVar_ = prevRunningVar + var;
   
    // L 
    MatrixXd L = xn_;
    for (int i = 0; i < L.rows(); ++i) {
        for (int j = 0; j < L.cols(); ++j) {
            L(i, j) *= g_(j);                                                                     
        }                                                                                                                               
    }

    // return L + b_
    MatrixXd R = L;
    for (int i = 0; i < R.cols(); ++i) {
        for (int j = 0; j < R.rows(); ++j) {
            R(j, i) +=  b_(i);
        }
    }

    return R;
}

MatrixXd BatchNormalization::backward(const MatrixXd& D) {
    // dbeta
    RowVectorXd dbeta = D.colwise().sum();

    // dgamma  
    RowVectorXd dgamma = (xn_.array() * D.array()).colwise().sum();

    // dxn
    MatrixXd dxn = D;
    for (int i = 0; i < dxn.rows(); ++i) {
        for (int j = 0; j < dxn.cols(); ++j) {
            dxn(i, j) *= g_(j);
        }
    }

    // dxc
    MatrixXd dxc = MatrixXd::Zero(dxn.rows(), dxn.cols());
    for (int i = 0; i < dxc.cols(); ++i) {
        for (int j = 0; j < dxc.rows(); ++j) {
            dxc(j, i) = dxn(j, i) / std_(i);
        }
    }

    // dstd 
    MatrixXd T2 = dxn.array() * xc_.array();
    RowVectorXd t3 = std_.array() * std_.array();
    MatrixXd T4 = MatrixXd::Zero(T2.rows(), T2.cols());
    for (int i = 0; i < T4.cols(); ++i) {
        for (int j = 0; j < T4.rows(); ++j) {
            T4(j, i) = T2(j, i) / t3(i);
        }
    }

    RowVectorXd dstd = T4.colwise().sum();
    dstd *= -1;

    // dvar
    RowVectorXd t5 = dstd * 0.5;
    RowVectorXd dvar = t5.array() / std_.array();

    // dxc
    MatrixXd T6 = xc_ * (2.0 / batchSize_);
    MatrixXd T7 = T6;
    for (int i = 0; i < T7.rows(); ++i) {
        for (int j = 0; j < T7.cols(); ++j) {
            T7(i, j) *= dvar(j);
        }
    }

    MatrixXd _dxc = dxc + T7;

    // dmu
    RowVectorXd dmu = _dxc.colwise().sum();

    // dx
    dmu *= (1.0 / batchSize_); 
    MatrixXd dx = _dxc;
    for (int i = 0; i < dx.cols(); ++i) {
        for (int j = 0; j < dx.rows(); ++j) {
            dx(j, i) -= dmu(i);
        }
    }

    dg_ = dgamma;
    db_ = dbeta;

    return dx;
}

Dropout::Dropout(double ratio) 
  : ratio_(ratio)
{
}

MatrixXd Dropout::forward(const MatrixXd& X, bool trainFlag) {
    if (trainFlag) {
        M_ = createMatrixXdRand(X.rows(), X.cols());
        for (int i = 0; i < M_.rows(); ++i) {
            for (int j = 0; j < M_.cols(); ++j) {
                if (M_(i, j) < ratio_) {
                    M_(i, j) = 0.0;
                } else {
                    M_(i, j) = 1.0;
                }
            }
        }
        return X.array() * M_.array();
    } 
    else {
        return X * (1.0 - ratio_);
    }
}

MatrixXd Dropout::backward(const MatrixXd& D) {
    return D.array() * M_.array();
}

