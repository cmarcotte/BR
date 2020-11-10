using DelimitedFiles
using Dierckx
using FileIO
using JLD2

fnames = readdir("$(r1)")
bifur = readdlm("$(r1)/$(fnames[1])");
label = readdlm("$(r1)/$(fnames[2])");
for n=3:(length(fnames)-2)
	
	auto_sol = readdlm("$(r1)/$(fnames[n])");
	
	x = vcat(auto_sol[:,1], 	auto_sol[2:end,1].+1.0	)
	y = vcat(auto_sol[:,2:end], 	auto_sol[2:end,2:end]	)
	
	spl = ParametricSpline(x.*label[n,12], transpose(y); k=5)
	
end

