set grid front
set xlabel "x"
set ylabel "y"
set format y "%.1f"
plot "./data/naive_SGD.txt" w lp t "SGD", "./data/naive_Momentum.txt" w lp t "Momentum", "./data/naive_AdaGrad.txt" w lp t "AdaGrad", "./data/naive_Adam.txt" w lp t "Adam"
pause -1
