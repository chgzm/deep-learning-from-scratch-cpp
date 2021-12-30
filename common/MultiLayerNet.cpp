#include "MultiLayerNet.h"
#include "Function.h"

MultiLayerNet::MultiLayerNet(
    std::size_t inputSize, 
    std::size_t hiddenLayerNum, 
    std::size_t hiddenSize, 
    std::size_t outputSize,
    WeightType weightType,
    double weight,
    double weightDecayLambda,
    std::vector<std::unique_ptr<Optimizer>>&& opt 
) : weightDecayLambda_(weightDecayLambda)
  , opts_(std::move(opt))
{
    A_.emplace_back(Affine(inputSize, hiddenSize));
    R_.emplace_back(Relu());
    for (std::size_t i = 1; i < hiddenLayerNum + 1; ++i) {
        if (i == hiddenLayerNum) {
            A_.emplace_back(Affine(hiddenSize, outputSize));
        } else {
            A_.emplace_back(Affine(hiddenSize, hiddenSize));
            R_.emplace_back(Relu());
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

void MultiLayerNet::gradient(const MatrixXd& X, const VectorXi& t) {
    this->loss(X, t, true);

    const MatrixXd X1 = S_.backward();
    MatrixXd X2 = A_[A_.size() - 1].backward(X1);  

    for (int i = A_.size() - 2; i >= 0; --i) {
        const MatrixXd X3 = R_[i].backward(X2);  
        X2 = A_[i].backward(X3);  
    }
}

MatrixXd MultiLayerNet::predict(const MatrixXd& X, bool trainFlag) {
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
            T2 = R_[i].forward(T1);
        }
    }

    return T1;
}

double MultiLayerNet::accuracy(const MatrixXd& X, const VectorXi& t) {
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

void MultiLayerNet::update() {
    for (std::size_t i = 0; i < A_.size(); ++i) {
        opts_[i]->updateRowVector(A_[i].b_, A_[i].db_);

        if (std::fabs(weightDecayLambda_) <= 0.0001) {
            opts_[i]->updateMatrix(A_[i].W_, A_[i].dW_);
        } else {
            const MatrixXd T = A_[i].W_ * weightDecayLambda_;
            const MatrixXd _dW = A_[i].dW_ + T;
            opts_[i]->updateMatrix(A_[i].W_, _dW);
        }

        if (opts_[i]->getType() == OptType::Adam) {
            ((Adam*)(opts_[i].get()))->increment();
        }
   }
}

double MultiLayerNet::loss(const MatrixXd& X, const VectorXi& t, bool trainFlag) {
    const MatrixXd Y = this->predict(X, trainFlag);

    if (std::fabs(weightDecayLambda_) <= 0.0001) {
        return S_.forward(Y, t);
    } else { 
        double weightDecay = 0.0;
        for (std::size_t i = 0; i < A_.size(); ++i) {
            MatrixXd T = A_[i].W_.unaryExpr([](double x) { return std::pow(x, 2); });
            const double sum = T.sum();

            weightDecay += 0.5 * weightDecayLambda_ * sum;
        }

        return S_.forward(Y, t) + weightDecay;
    } 
}
