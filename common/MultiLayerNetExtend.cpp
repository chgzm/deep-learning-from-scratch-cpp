#include "MultiLayerNetExtend.h"
#include "Function.h"
#include "Debug.h"

MultiLayerNetExtend::MultiLayerNetExtend(
    std::size_t inputSize, 
    std::size_t hiddenLayerNum, 
    std::size_t hiddenSize, 
    std::size_t outputSize,
    WeightType weightType,
    double weight,
    bool useDropout,
    double dropoutRatio,
    std::vector<std::unique_ptr<Optimizer>>&& opt 
) : opts_(std::move(opt))
  , useDropout_(useDropout)
{
    A_.emplace_back(Affine(inputSize, hiddenSize));
    R_.emplace_back(Relu());
    B_.emplace_back(BatchNormalization(hiddenSize, hiddenSize, 0.9));
    if (useDropout_) {
        D_.emplace_back(Dropout(dropoutRatio));
    }

    for (std::size_t i = 1; i < hiddenLayerNum + 1; ++i) {
        if (i == hiddenLayerNum) {
            A_.emplace_back(Affine(hiddenSize, outputSize));
        } else {
            A_.emplace_back(Affine(hiddenSize, hiddenSize));
            R_.emplace_back(Relu());
            B_.emplace_back(BatchNormalization(hiddenSize, hiddenSize, 0.9));
            if (useDropout_) {
                D_.emplace_back(Dropout(dropoutRatio));
            }
        }
    }

    S_ = SoftmaxWithLoss();

    switch (weightType) {
    case WeightType::He: {
        for (std::size_t i = 0; i < hiddenLayerNum + 1; ++i) {
            const double scale = (i == 0) ? (std::sqrt(2.0 / inputSize)) : (std::sqrt(2.0 / hiddenSize));
            A_[i].W_ *= scale;
        }
        break;
    }
    case WeightType::Xavier: {
        for (std::size_t i = 0; i < hiddenLayerNum + 1; ++i) {
            const double scale = (i == 0) ? (std::sqrt(1.0 / inputSize)) : (std::sqrt(1.0 / hiddenSize));
            A_[i].W_ *= scale;
        }
        break;
    }
    case WeightType::STD: {
        for (std::size_t i = 0; i < hiddenLayerNum + 1; ++i) {
            A_[i].W_ *= weight;
        }
        break;
    }
    default: {
        break;
    }
    }
}

void MultiLayerNetExtend::gradient(const MatrixXd& X, const VectorXi& t) {
    this->loss(X, t, true);

    const MatrixXd X1 = S_.backward();
    MatrixXd X2 = A_[A_.size() - 1].backward(X1);  

    for (int i = A_.size() - 2; i >= 0; --i) {
        if (useDropout_) {
            X2 = D_[i].backward(X2);
        }

        X2 = R_[i].backward(X2);  
        X2 = B_[i].backward(X2);
        X2 = A_[i].backward(X2);  
    }
}

MatrixXd MultiLayerNetExtend::predict(const MatrixXd& X, bool trainFlag) {
    MatrixXd T1, T2;
    for (std::size_t i = 0; i < A_.size(); ++i) {
        if (i == 0) {
            T1 = A_[i].forward(X);
        } else {
            T1 = A_[i].forward(T2);
        }

        if (i == (A_.size() - 1)) {
            break;
        } else {
            T2 = B_[i].forward(T1);
            T2 = R_[i].forward(T2);
            if (useDropout_) {
                T2 = D_[i].forward(T2, trainFlag);
            }
        }
    }

    return T1;
}

double MultiLayerNetExtend::accuracy(const MatrixXd& X, const VectorXi& t) {
    const MatrixXd Y = this->predict(X, false);
    std::size_t cnt = 0;
    const RowVectorXi y = argmax(Y);
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) == t(i)) {
            ++cnt;
        }
    }

    return (double)cnt / X.rows();
}

void MultiLayerNetExtend::update() {
    for (std::size_t i = 0; i < A_.size(); ++i) {
        opts_[i]->updateRowVector(A_[i].b_, A_[i].db_);
        opts_[i]->updateMatrix(A_[i].W_, A_[i].dW_);

        if (opts_[i]->getType() == OptType::Adam) {
            ((Adam*)(opts_[i].get()))->increment();
        }
    }

    for (std::size_t i = 0; i < B_.size(); ++i) { 
        opts_[A_.size() + i]->updateRowVector(B_[i].g_, B_[i].dg_);
        opts_[A_.size() + B_.size() + i]->updateRowVector(B_[i].b_, B_[i].db_);
    }
}

double MultiLayerNetExtend::loss(const MatrixXd& X, const VectorXi& t, bool trainFlag) {
    const MatrixXd Y = this->predict(X, trainFlag);
    return S_.forward(Y, t);
}
