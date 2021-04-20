using DelimitedFiles, FileIO
using DynamicalSystems
using Random, Statistics
using PyPlot

function dolyap(x; ks=1:100, mm=1:3, DD=1:5:100, tt=1:5:100, Δt = 1)
	
	di = Cityblock()
	lyap = []
	for m in mm, D in DD, tau in tt
		try
			if D==1 && tau>1
				#skip
			else
				R = embed(x, D, tau)
				if size(R,2) <= maximum(ks)
					ks = ks[1:end÷2]
				end
				E = numericallyapunov(R, ks; distance = di, ntype = NeighborNumber(m))
				#E = numericallyapunov(R, ks; distance = di, ntype = WithinRange(m*std(x)/(maximum(x)-minimum(x))))
				
				# gives the linear slope, i.e. the Lyapunov exponent
				λ = linear_region(ks.*Δt, E)[2]
				push!(lyap,λ)
			end
		catch
		end
	end

	return lyap
end

function fit_sequence(x)

	s = getasymptoticperiodicity(x)	
	
	#fig = plt.figure()
	y = abs.(x.-x[end])
	#plt.semilogy(y, "o-C0", label="$xname\$_n\$ - $xname\$_\\infty\$")
	
	ii=1:s:length(y)
	inds, ll = linear_region(ii, log.(y[ii]));
	
	#plt.semilogy((ii[inds[1]:inds[end]]).-1, y[ii[inds[1]:inds[end]]], "x:C2", label="linear region")
	#yl = plt.ylim()
	
	#plt.semilogy(1:length(y), exp.(ll.*(1:length(y))), "--k", label="\$ \\exp($(round(ll,digits=3)) k)\$")
	
	#plt.xlabel("\$ k \$")
	#plt.ylim(yl)
	#plt.legend(loc="best", edgecolor="none")
	#if saveplt; plt.savefig("./pltz/$(xname)_$(fig_append)_truth.svg", bbox_inches="tight"); end
	#plt.close()
	return inds, ll

end

function mostpopularelement(x::Array{Int,1})
	#=sort!(x)
	popelem	= x[1]
	maxcount 	= 1
	curcount	= 1
	for n=2:length(x)
		if x[n] == x[n-1]
			curcount = curcount + 1
		elseif curcount > maxcount
			maxcount = curcount
		end
		popelem = x[n - 1]
              	curr_count = 1
	end
	# If last element is most frequent
	if curcount > maxcount
		max_count = curr_count
		popelem = x[n - 1]
	end
	
	return popelem,maxcount
	=#
	maxcount = 0
	popelem  = nothing
	for X in x
		I = findall(X .== x)
		if length(I) > maxcount
			popelem = X
			maxcount = length(I)
		end
	end
	return popelem, maxcount
end

function getasymptoticperiodicity(x)

	y = abs.(x.-x[end])
	miny = y[1]
	ind = [1]
	for n=2:(length(y)-1)
		if y[n] < miny
			push!(ind, n)
			miny = y[n]
		end
	end
	
	# now we select the most popular element in diff(ind)
	s, p = mostpopularelement(diff(ind))
	
	return s
end

plt.style.use("seaborn-paper")
fig,axs = plt.subplots(3, 1, figsize=(5,6), sharex=true, constrained_layout=true)
K = []

for n=400:-1:40
	dat = readdlm("./tisean/dats/APDs_$(n).dat")[:]
	
	Kc = [testchaos01(dat .+ (0.0).*mean(dat).*(2.0.*rand(Float64, length(dat)).-1.0), c, length(dat)÷10) for c in (3π/5).*rand(Float64,2^13) .+ π/4]
	push!(K, Kc)
	
	λ = dolyap(dat; mm=2, DD=100:1:200, tt=1)
	#λ = λ[(end-25):end]
	
	inds, ll = fit_sequence(dat); 
	
	axs[1].plot(n.*ones(size(dat)), dat, ".k", linestyle="none", alpha=0.3)
	axs[2].violinplot(K[end], [n], showmeans=true, showextrema=true, showmedians=true)
	axs[3].plot(n.*ones(size(λ)), λ, ".k", linestyle="none", alpha=0.3)
	axs[3].violinplot(λ, [n], showmeans=true, showextrema=true, showmedians=true)
	axs[3].plot(n, ll, "or", linestyle="none")
end
axs[3].set_xlabel("BCL [ms]")
axs[3].set_ylabel("\$ \\lambda \$ [ms\$^{-1}\$]")
axs[2].set_ylabel("\$ K_c \$")
axs[1].set_ylabel("APD [ms]")
plt.savefig("./fig.svg", bbox_inches="tight")
plt.close()

