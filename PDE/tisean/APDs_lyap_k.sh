dir=/home/chris/Development/Julia/BR/PDE/tisean

rm fit.log

for ((n=400 ; n >= 40; n--)); do
	/home/chris/bin/lyap_k -m 1 -M 100 -d 1 -n 15 -r 1.0 -R 15.0 -o ${dir}/lyaps/APDs_${n}.lyap ${dir}/dats/APDs_${n}.dat
	gnuplot -c ${dir}/tiseanplot.gp "${dir}/lyaps/APDs_${n}.lyap" "${dir}/plots/APDs_${n}.pdf"
done
