#!/usr/local/bin/gnuplot --persist
set term pdfcairo size 4, 3
set output ARG3
a=1.0
b=-6.0
g(x) = a*x + b
fit [0:4] g(x) '/home/chris/Development/Julia/BR/map/map_APDs.txt.lyap' using 1:2 via a, b
set title sprintf("%f * x + %f",a,b)
plot '/home/chris/Development/Julia/BR/map/map_APDs.txt.lyap' with lines, g(x)
