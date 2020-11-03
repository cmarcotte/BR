dir=/home/chris/Development/Julia/BR/map/

for c in 1 2 3 4 5 6 7 8 9 10; do
	for l in 50 100 500 1000 4000; do
		./lyap_k -m 1 -M 3 -d 1 -R 5.0 -n 15 -s 10 -c $c -l 50 ${dir}/map_APDs.txt
		gnuplot -c ${dir}/tiseanplot.gp ${c} ${l} "${dir}/lyap_${c}_${l}.pdf"
	done
done
