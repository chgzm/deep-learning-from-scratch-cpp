#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Matrix.h"

enum class OptType : uint8_t {
    None, 
    SGD,
    Momentum,
    AdaGrad,
    Adam
};

class Optimizer {
public:
    Optimizer(OptType type);
    virtual ~Optimizer() {}

    inline OptType getType() const {
        return type_;
    }

    virtual void updateRowVector(RowVectorXd& x, const RowVectorXd& dx) = 0;
    virtual void updateMatrix(MatrixXd& X, const MatrixXd& dX) = 0;

private:
    OptType type_;
};

class SGD : public Optimizer {
public:
    SGD(double lr);
    ~SGD() = default;

    void updateRowVector(RowVectorXd& x, const RowVectorXd& dx) override;
    void updateMatrix(MatrixXd& X, const MatrixXd& dX) override;

private:
    double lr_;
};

class Momentum : public Optimizer {
public:
    Momentum(double lr, double momentum);
    ~Momentum() = default;

    void updateRowVector(RowVectorXd& x, const RowVectorXd& dx) override;
    void updateMatrix(MatrixXd& X, const MatrixXd& dX) override;

private:
    bool vInit_ = false;
    bool mInit_ = false;
    double lr_;
    double momentum_;
    RowVectorXd vv_;
    MatrixXd mv_;
};

class AdaGrad : public Optimizer {
public:
    AdaGrad(double lr);
    ~AdaGrad() = default;

    void updateRowVector(RowVectorXd& x, const RowVectorXd& dx) override;
    void updateMatrix(MatrixXd& X, const MatrixXd& dX) override;

private:
    bool vInit_ = false;
    bool mInit_ = false;
    double lr_;
    RowVectorXd vh_;
    MatrixXd mh_;
};

class Adam : public Optimizer {
public:
    Adam(double lr, double beta1, double beta2);
    ~Adam() = default;

    void updateRowVector(RowVectorXd& x, const RowVectorXd& dx) override;
    void updateMatrix(MatrixXd& X, const MatrixXd& dX) override;
    
    void increment();

private:
    bool vInit_ = false;
    bool mInit_ = false;
    double lr_;
    double beta1_;
    double beta2_;
    std::size_t iter_ = 1;
    RowVectorXd vm_;
    RowVectorXd vv_;
    MatrixXd mm_;
    MatrixXd mv_;
};

#endif
