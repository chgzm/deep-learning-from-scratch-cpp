#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <Matrix.h>
#include <Function.h>
#include <Gnuplot.h>
#include <Optimizer.h>

static double df_x(double x) {
    return x / 10.0;
}

static double df_y(double y) {
    return 2.0 * y;
}

static void run(Optimizer& optimizer) {
    RowVectorXd x(2);
    x << -7.0, 2.0;
    RowVectorXd dx = RowVectorXd::Zero(2);

    FILE* fp = nullptr;
    switch (optimizer.getType()) {
    case OptType::SGD: {
        fp = std::fopen("./data/naive_SGD.txt", "w");
        break;
    }
    case OptType::Momentum: {
        fp = std::fopen("./data/naive_Momentum.txt", "w");
        break;
    }
    case OptType::AdaGrad: {
        fp = std::fopen("./data/naive_AdaGrad.txt", "w");
        break;
    }
    case OptType::Adam: {
        fp = std::fopen("./data/naive_Adam.txt", "w");
        break;
    }
    default: {
        break;
    }
    }

    if (!fp) {
        std::fprintf(stderr, "Failed to open file.\n");
        return;
    }

    for (int i = 0; i < 30; ++i) {
        std::fprintf(fp, "%lf %lf\n", x(0), x(1));
        
        dx(0) = df_x(x(0));
        dx(1) = df_y(x(1));

        optimizer.updateRowVector(x, dx);
        if (optimizer.getType() == OptType::Adam) {
            ((Adam&)(optimizer)).increment();
        }
    }

    std::fclose(fp);
}

int main() {
    SGD sgd(0.95); 
    run(sgd); 

    Momentum momentum(0.1, 0.9);
    run(momentum);

    AdaGrad adaGrad(1.5);
    run(adaGrad);

    Adam adam(0.3, 0.9, 0.999);
    run(adam); 

    plotGpFile("./plot_optimization_naive.gp");

    return 0;   
}
