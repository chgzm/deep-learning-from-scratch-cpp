#include <Matrix.h>
#include <Function.h>
#include <Gnuplot.h>

constexpr std::size_t NUM_DATA = 1000; 
constexpr std::size_t NUM_FEATURES = 100; 
constexpr std::size_t NUM_PARAMS = NUM_DATA * NUM_FEATURES;
constexpr std::size_t NODE_NUM = 100;
constexpr std::size_t HIDDEN_LAYER_SIZE = 5;

static void storeActivation(const MatrixXd& X, double activations[HIDDEN_LAYER_SIZE][NUM_PARAMS], std::size_t idx) {
    std::size_t pos = 0;
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            activations[idx][pos] = X(i, j);
            ++pos;
        }
    }
}

static void writeActivations(double activations[HIDDEN_LAYER_SIZE][NUM_PARAMS]) {
    FILE* fp = std::fopen("./data/weight_init_activation_histogram.txt", "w");
    if (!fp) {
        std::fprintf(stderr, "Failed to open file.\n");
        return;
    }
    
    for (std::size_t i = 0; i < NUM_PARAMS; ++i) {
        std::fprintf(fp, "%lf %lf %lf %lf %lf\n", activations[0][i], activations[1][i], activations[2][i], activations[3][i], activations[4][i]);
    }

    std::fclose(fp);
}

int main() {
    const MatrixXd X = MatrixXd::Random(NUM_DATA, NUM_FEATURES);

    double activations[HIDDEN_LAYER_SIZE][NUM_PARAMS];
    for (std::size_t i = 0; i < HIDDEN_LAYER_SIZE; ++i) {
        const MatrixXd W = MatrixXd::Random(NODE_NUM, NODE_NUM);
        const MatrixXd A = X * W;
        const MatrixXd Y = sigmoid(A);

        storeActivation(Y, activations, i);
    }

    writeActivations(activations);

    plotGpFile("plot_weight_init_activation_histogram.gp");

    return 0;   
}
