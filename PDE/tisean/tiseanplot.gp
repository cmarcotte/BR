#!/usr/local/bin/gnuplot --persist
set term pdfcairo size 4, 3
set output ARG2
a=0.0
b=0.0
g(x) = a*x + b
fit [0:3] g(x) ARG1 using 1:2 via a, b
set title sprintf("%f * x + %f",a,b)
plot ARG1 with lines, g(x)
