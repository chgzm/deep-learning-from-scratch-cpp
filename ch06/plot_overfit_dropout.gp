set grid front
set xlabel "epochs"
set ylabel "accuracy"
set format y "%.1f"
plot "./data/overfit_dropout.txt" u 1:2 w l lw 2 t "train", "./data/overfit_dropout.txt" u 1:3 w l lw 2 t "test"
pause -1
