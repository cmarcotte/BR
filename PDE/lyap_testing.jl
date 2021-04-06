using DynamicalSystems
using DelimitedFiles
using Statistics
using PyPlot
using FileIO
using JLD2

savefile = "./data/data_100.0_500.0.jld2"

APDs = load(savefile,"APDs")
APAs = load(savefile,"APAs")
DIs  = load(savefile,"DIs" )
BCLs = load(savefile,"BCLs")
	
#=	
function fit_sequence(x, xname, fig_append; s=1, saveplt=true)
	fig = plt.figure()
	y = abs.(x.-x[end])
	plt.semilogy(y, "o-C0", label="$xname\$_n\$ - $xname\$_\\infty\$")
	
	ii=1:s:length(y)
	inds, ll = linear_region(ii, log.(y[ii]));
	
	plt.semilogy((ii[inds[1]:inds[end]]).-1, y[ii[inds[1]:inds[end]]], "x:C2", label="linear region")
	yl = plt.ylim()
	
	plt.semilogy(1:length(y), exp.(ll.*(1:length(y))), "--k", label="\$ \\exp($(round(ll,digits=3)) k)\$")
	
	plt.xlabel("\$ k \$")
	plt.ylim(yl)
	plt.legend(loc="best", edgecolor="none")
	if saveplt; plt.savefig("./pltz/$(xname)_$(fig_append)_truth.svg", bbox_inches="tight"); end
	plt.close()
	return inds, ll
end


INDS = Array{Any}(undef,length(APDs))
lyap = Array{Any}(undef,length(APDs))
Ss   = Array{Any}(undef,length(APDs))
for n=1:length(APDs)
	inds, ll = fit_sequence(APDs[n], "APD", "n_$(n)"; s=1); 
	INDS[n] = inds
	lyap[n] = ll 
	Ss[n] = 1
end
for n in vcat(95:112, 245:267, 298:316, 323:361)
	inds, ll = fit_sequence(APDs[n], "APD", "n_$(n)"; s=2); 
	INDS[n] = inds
	lyap[n] = ll 
	Ss[n] = 2
end
=#


function dolyap(x; ks=1:100, mm=1:3, DD=1:10, tt=1:10, Δt = 1)
	fig,axs = plt.subplots(2,2,figsize=(8,8), sharey="row", sharex="row")
	for (i, di) in enumerate([Euclidean(), Cityblock()])
		lyap = []
		for m in mm, D in DD, tau in tt
			try
				if D==1 && tau>1
					#skip
				else
					R = embed(x, D, tau)
					E = numericallyapunov(R, ks; distance = di, ntype = NeighborNumber(m))
					#E = numericallyapunov(R, ks; distance = di, ntype = WithinRange(m*std(x)/(maximum(x)-minimum(x))))
					λ = linear_region(ks.*Δt, E)[2]
					push!(lyap,λ)
					# gives the linear slope, i.e. the Lyapunov exponent
					axs[1,i].plot(ks .- 1, E .- E[1], color="C$(m)", label = "m=$m, D=$D, tau=$tau, λ=$(round(λ, digits = 3))")
					global mean_l = round(mean(lyap),	digits=3)
					global min_l  = round(minimum(lyap), 	digits=3)
					global max_l  = round(maximum(lyap), 	digits=3)
					global var_l  = round(var(lyap), 	digits=3)
					axs[1,i].set_title("\$ \\lambda \\in [$min_l, $max_l] \\sim \\mathcal{N}($mean_l, $var_l)\$")
					#axs[i].legend()
					tight_layout()
				end
			catch
			end
		end
		axs[2,i].hist(lyap, bins="auto", density=true)
		yl=axs[1,i].get_ylim()
		axs[1,i].set_xlabel("\$ k \$")
		axs[1,1].set_ylabel("\$ S(k) \$")
		axs[2,i].set_xlabel("\$ \\lambda \$")
		axs[2,1].set_ylabel("\$ P(\\lambda) \$")
		axs[2,i].set_yscale("log")
		axs[1,i].plot((ks.-1), min_l.*ks, "--r")
		axs[1,i].plot((ks.-1), max_l.*ks, "--r")
		axs[1,i].plot((ks.-1), mean_l.*ks, "-k")
		axs[1,i].plot((ks.-1), (mean_l + 0.5sqrt(var_l)).*ks, "--k")
		axs[1,i].plot((ks.-1), (mean_l - 0.5sqrt(var_l)).*ks, "--k")
		axs[1,i].plot((ks.-1), (mean_l + sqrt(var_l)).*ks, "-.k")
		axs[1,i].plot((ks.-1), (mean_l - sqrt(var_l)).*ks, "-.k")
		axs[1,i].plot((ks.-1), (mean_l + 2sqrt(var_l)).*ks, ":k")
		axs[1,i].plot((ks.-1), (mean_l - 2sqrt(var_l)).*ks, ":k")
		axs[1,i].set_ylim(yl)
	end
end


