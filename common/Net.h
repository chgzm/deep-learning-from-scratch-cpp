#ifndef NET_H
#define NET_H

#include "Matrix.h"

class Net {
public:
    Net() = default;
    virtual ~Net() {}

    virtual void gradient(const MatrixXd& X, const VectorXi& t) = 0;
    virtual void update() = 0;
    virtual double accuracy(const MatrixXd& X, const VectorXi& t) = 0;
    virtual double loss(const MatrixXd& X, const VectorXi& t, bool trainFlg) = 0;
};

#endif
