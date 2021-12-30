## deep-learning-from-scratch-cpp
Implement the sample sources in deep-learning-from-scratch ([ゼロから作る Deep Learning](https://github.com/oreilly-japan/deep-learning-from-scratch)) in C++.

## Install dependency
Eigen:

```bash
$ sudo apt install libeigen3-dev
```

You also need gnuplot to draw graphs and GoogleTest to run tests.

## Build and run
Go to the folder for each chapter and execute `make` , and run binary.

Example:

```bash
$ cd ch05
$ make
$ ./TrainNeuralNet
train acc, test acc | 0.153883, 0.153900
train acc, test acc | 0.899217, 0.903400
train acc, test acc | 0.923717, 0.926800
train acc, test acc | 0.935000, 0.933600
train acc, test acc | 0.944083, 0.940300
train acc, test acc | 0.953017, 0.949800
train acc, test acc | 0.957133, 0.953600
train acc, test acc | 0.960833, 0.956500
...
```
