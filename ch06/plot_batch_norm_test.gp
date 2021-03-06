set terminal qt size 1200, 900
set grid front
set style fill solid
set yrange [0:1]
set format y "%.1f"
set multiplot layout 4,4

set size 0.25, 0.25
set title "W:1.0"
plot "./data/batch_norm_test_1.0000.txt" u 1:2  w l t "", "./data/batch_norm_test_1.0000.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.5412"
plot "./data/batch_norm_test_0.5412.txt" u 1:2  w l t "", "./data/batch_norm_test_0.5412.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.2929"
plot "./data/batch_norm_test_0.2929.txt" u 1:2  w l t "", "./data/batch_norm_test_0.2929.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.1585"
plot "./data/batch_norm_test_0.1585.txt" u 1:2  w l t "", "./data/batch_norm_test_0.1585.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0858"
plot "./data/batch_norm_test_0.0858.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0858.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0464"
plot "./data/batch_norm_test_0.0464.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0464.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0251"
plot "./data/batch_norm_test_0.0251.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0251.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0136"
plot "./data/batch_norm_test_0.0136.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0136.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0074"
plot "./data/batch_norm_test_0.0074.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0074.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0040"
plot "./data/batch_norm_test_0.0040.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0040.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0022"
plot "./data/batch_norm_test_0.0022.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0022.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0012"
plot "./data/batch_norm_test_0.0012.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0012.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0006"
plot "./data/batch_norm_test_0.0006.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0006.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0003"
plot "./data/batch_norm_test_0.0003.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0003.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0002"
plot "./data/batch_norm_test_0.0002.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0002.txt" u 1:3  w l t ""
set size 0.25, 0.25
set title "W:0.0001"
plot "./data/batch_norm_test_0.0001.txt" u 1:2  w l t "", "./data/batch_norm_test_0.0001.txt" u 1:3  w l t ""

pause -1
