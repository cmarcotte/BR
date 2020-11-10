# from bash:
# > conda activate base
julia make_dats.jl 1 2.3 1.05 1.0
julia make_dats.jl 2 2.3 3.12 1.0
julia make_dats.jl 4 2.3 4.95 1.0
julia make_dats.jl 8 2.3 5.25 1.0 # 5.20 < f < 6.60
julia make_dats.jl 8 2.3 6.25 1.0 # 5.20 < f < 6.60

auto br.auto
for n in 1 2 3 4 5; do
	mkdir ./r${n}
	autox to_matlab_sh.xauto r${n} ./r${n}/
done


