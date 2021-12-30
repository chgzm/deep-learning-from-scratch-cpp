#include "Optimizer.h"
#include "Matrix.h"
#include <type_traits>

Optimizer::Optimizer(OptType type) 
  : type_(type)
{
}

SGD::SGD(double lr) 
  : Optimizer(OptType::SGD),
    lr_(lr)
{
}

void SGD::updateRowVector(RowVectorXd& x, const RowVectorXd& dx) {
    x -= lr_ * dx;
}

void SGD::updateMatrix(MatrixXd& X, const MatrixXd& dX) {
    X -= lr_ * dX;
}

Momentum::Momentum(double lr, double momentum) 
  : Optimizer(OptType::Momentum),
    lr_(lr),
    momentum_(momentum)
{
}

void Momentum::updateRowVector(RowVectorXd& x, const RowVectorXd& dx) {
    if (!vInit_) {
        vv_ = RowVectorXd::Zero(x.size());
        vInit_ = true;
    }

    vv_ = momentum_ * vv_ - lr_ * dx;
    x += vv_;
}

void Momentum::updateMatrix(MatrixXd& X, const MatrixXd& dX) {
    if (!mInit_) {
        mv_ = MatrixXd::Zero(X.rows(), X.cols());
        mInit_ = true;
    }

    mv_ = momentum_ * mv_ - lr_ * dX;
    X += mv_;
}

AdaGrad::AdaGrad(double lr) 
  : Optimizer(OptType::AdaGrad),
    lr_(lr)
{
}

void AdaGrad::updateRowVector(RowVectorXd& x, const RowVectorXd& dx) {
    if (!vInit_) {
        vh_ = RowVectorXd::Zero(x.size());
        vInit_ = true;
    }
        
    RowVectorXd buf1 = dx.array() * dx.array();
    vh_ += buf1;
    RowVectorXd buf2 = vh_.unaryExpr([](double x) { return std::sqrt(x) + 1e-7; });
    for (int i = 0; i < x.size(); ++i) {
        x(i) -= lr_ * (dx(i) / buf2(i));
    }    
}

void AdaGrad::updateMatrix(MatrixXd& X, const MatrixXd& dX){
    if (!mInit_) {
        mh_ = MatrixXd::Zero(X.rows(), X.cols());
        mInit_ = true;
    }
    
    MatrixXd buf1 = dX.array() * dX.array();
    mh_ += buf1;
    MatrixXd buf2 = mh_.unaryExpr([](double x) { return std::sqrt(x) + 1e-7; });

    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            X(i, j) -= lr_ * (dX(i, j) / buf2(i, j));
        }
    }
}

Adam::Adam(double lr, double beta1, double beta2) 
  : Optimizer(OptType::Adam),
    lr_(lr),
    beta1_(beta1),
    beta2_(beta2)
{
}

void Adam::updateRowVector(RowVectorXd& x, const RowVectorXd& dx){
    if (!vInit_) {
        vm_ = RowVectorXd::Zero(x.size());
        vv_ = RowVectorXd::Zero(x.size());
        vInit_ = true; 
    }

    const double lrTmp = lr_ * std::sqrt(1.0 - std::pow(beta2_, iter_ + 1)) / (1.0 - std::pow(beta1_, iter_ + 1)); 
    vm_ += (1 - beta1_) * (dx - vm_);
    vv_ += (1 - beta2_) * (dx.unaryExpr([](double x) { return std::pow(x, 2); }) - vv_);

    RowVectorXd buf = vv_.unaryExpr([](double x) { return std::sqrt(x) + 1e-7; });
    for (int i = 0; i < x.size(); ++i) {
        x(i) -= lrTmp * vm_(i) / buf(i);
    }
}   

void Adam::updateMatrix(MatrixXd& X, const MatrixXd& dX){
    if (!mInit_) {
        mm_ = MatrixXd::Zero(X.rows(), X.cols());
        mv_ = MatrixXd::Zero(X.rows(), X.cols());
        mInit_ = true;
    }

    const double lrTmp = lr_ * std::sqrt(1.0 - std::pow(beta2_, iter_ + 1)) / (1.0 - std::pow(beta1_, iter_ + 1)); 
    mm_ += (1 - beta1_) * (dX - mm_);
    mv_ += (1 - beta2_) * (dX.unaryExpr([](double x) { return std::pow(x, 2); }) - mv_);

    MatrixXd buf = mv_.unaryExpr([](double x) { return std::sqrt(x) + 1e-7; });
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            X(i, j) -= lrTmp * mm_(i, j) / buf(i, j);
        }
    }
}

void Adam::increment() {
    ++iter_;
}
