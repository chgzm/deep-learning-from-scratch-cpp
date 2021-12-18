#include <cstdio>
#include <cstdlib>
#include <MNIST.h>
#include <Matrix.h>
#include <Function.h>
#include <Eigen/Dense>

int main() {
    MNIST mnist;
    if (mnist.load("./../dataset/t10k-images-idx3-ubyte", "./../dataset/t10k-labels-idx1-ubyte") != 0) {
        fprintf(stderr, "Failed to load MNIST.\n");
        return -1;
    }

    MatrixXd W1(784, 50);
    if (initMatrix(W1, "./data/W1.csv") != 0) {
        fprintf(stderr, "Failed to init W1\n");
        return -1;
    }

    MatrixXd W2(50, 100);
    if (initMatrix(W2, "./data/W2.csv") != 0) {
        fprintf(stderr, "Failed to init W2\n");
        return -1;
    }

    MatrixXd W3(100, 10);
    if (initMatrix(W3, "./data/W3.csv") != 0) {
        fprintf(stderr, "Failed to init W3\n");
        return -1;
    }

    RowVectorXd b1(50);
    if (initRowVector(b1, "./data/b1.csv") != 0) {
        fprintf(stderr, "Failed to init b1\n");
        return -1;
    }

    RowVectorXd b2(100);
    if (initRowVector(b2, "./data/b2.csv") != 0) { 
        fprintf(stderr, "Failed to init b2\n");
        return -1;
    }

    RowVectorXd b3(10);
    if (initRowVector(b3, "./data/b3.csv") != 0) {
        fprintf(stderr, "Failed to init b3\n");
        return -1;
    }

    int accuracyCnt = 0;
    const MatrixXd& images = mnist.getImages();
    const VectorXi& labels = mnist.getLabels();
    for (int i = 0; i < images.rows(); ++i) {
        const MatrixXd& M = mnist.getImages();
        const RowVectorXd& x = M.row(i);
        const RowVectorXd a1 = x * W1 + b1;
        const RowVectorXd z1 = sigmoid(a1);
        const RowVectorXd a2 = z1 * W2 + b2;
        const RowVectorXd z2 = sigmoid(a2);
        const RowVectorXd a3 = z2 * W3 + b3;
        const RowVectorXd y = softmax(a3);

        const int p = argmax(y);
        if (p == labels(i)) {
            ++accuracyCnt;
        }
    }

    std::printf("Accuracy:%lf\n", (double)(accuracyCnt) / images.rows());

    return 0;
}
