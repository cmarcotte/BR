using DifferentialEquations
using DynamicalSystems
using Statistics
using Printf
using PyPlot
using JLD2, FileIO

# bifurcation_run.jl and BR.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing birfurcation_run.jl and BR.jl>
#	>> julia bifurcation_run.jl
push!(LOAD_PATH,pwd())
using BR
BR.test()

plt.style.use("seaborn-paper")

# initial condition
u0 = BR.u0

# parameters
p = BR.p
p[2] = 2.3; p[4] = 1.0

# define V90 threshold
V90 = BR.V90

# get runprob from BR
prob = BR.prob
runprob = BR.runprob

# new tspan
tspan = [0.0, 40000.0]

# collect APDs
APDs = []

# collect APAs
APAs = []

# collect DIs
DIs = []

# collect periods
BCLs = []

# collect Lyapunov Exponents
ALEs = []
DLEs = []

# collect dimension estimates
ADEs = []
DDEs = []

# BCL sweep range
BCL_range = 1000.0:-2.0:40.0

filename = "zoom_$(p[2])_$(p[4])"
savefile = "./data/data_$(filename).jld2"

function compute_lyap_dim(data; plot_stretch=false)

	lyap = []
	dim  = []
	
	if plot_stretch
		fig = figure(figsize=(10,6))
	end
	
	ks = 1:20
	
	for (i, di) in enumerate([Euclidean(), Cityblock()])
		if plot_stretch
			subplot(1, 2, i)
			title("Distance: $(di)", size = 18)
		end
		neigh=1:5
		for n in neigh
			for D in 1:4
				R = embed(data, D, 1)
				d = takens_best_estimate(R, std(V)/4)
				E = numericallyapunov(R, ks; distance = di, ntype=FixedMassNeighborhood(n))
				l = linear_region(ks.*1, E)[2]
				if plot_stretch
					plot(ks .- ks[1], E .- E[1], label = "D=$D, n=$n, λ=$(round(l, digits = 3))")
					legend(ncol=length(neigh), loc="lower center")
					tight_layout()
				end
				push!(dim, d)
				push!(lyap, l)
			end
		end
	end
	return lyap, dim
end

try
	global APDs = load(savefile,"APDs")
	global APAs = load(savefile,"APAs")
	global DIs  = load(savefile,"DIs" )
	global BCLs = load(savefile,"BCLs")
	global ALEs = load(savefile,"ALEs")
	global DLEs = load(savefile,"DLEs")
	global DDEs = load(savefile,"DDEs")
	global ADEs = load(savefile,"ADEs")
	
	print("Loaded data from $(savefile)... skipping compute.\n")
catch
	print("Could not load data from $(savefile)... computing:\n")

	# sweep BCLs:
	for BCL in BCL_range

		# make sure the forcing frequency is passed to the model correctly 
		# depending on magnitude and parity of p[4]
		f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;

		# write out the progress
		println("Running BCL=$(BCL).\n")
		
		# remake prob with new u0, p, and tspan and solve
		sol = runprob(prob,u0,p,tspan)
		
		# plot sol
		if isapprox(BCL,1000.0) || isapprox(BCL,400.0) || isapprox(BCL, 130.0)
			BR.pltsol(sol,p)
			hist(diff(sol.t[:]),bins="auto",density=true,histtype="step")
			xscale("log")
			yscale("log")
			xlabel("\$ h_t \$")
			ylabel("\$ P(h_t) \$")
			savefig("./figures/dt_hist.pdf")
			close()
		end

		# extract the time and voltage traces
		t = sol.t[:]
		V = sol[1,:]
		
		# storage for APD, DI, APA for this BCL
		APD = Float64[]
		DI  = Float64[]
		APA = Float64[]

		# accumulators for APD, DI, APA for this BCL
		apd = 0.0
		di  = 0.0
		apa = V90

		# look over the voltage trace, comparing to V90
		for n in 2:length(t)-1
		
			if V[n] > V90
				if V[n-1] < V90
					# V(t[n-1] + t90) == V90 + o((t[n]-t[n-1])^2)
					#t90 = ((V90-V[n-1])/(V[n]-V[n-1]))*(t[n]-t[n-1])
					#apd = t90;
					apd = 0.0;
					apa = V[n];
				end
				apd = apd + (t[n]-t[n-1]);
				apa = max(apa, V[n])
				if V[n+1] < V90
					# V(t[n] + t90) == V90 + o((t[n+1]-t[n])^2)
					#t90 = ((V[n]-V90)/(V[n]-V[n+1]))*(t[n+1]-t[n])
					#apd = apd + t90
					push!(APD, apd)
					push!(APA, apa)
					apd = 0.0;
					apa = V90;
				end
			end
			
			if V[n] < V90
				if V[n-1] > V90
					#V(t[n-1]+t90) == V90 + o((t[n]-t[n-1])^2)
					#t90 = ((V[n-1]-V90)/(V[n-1]-V[n]))*(t[n]-t[n-1])
					#di = t[n]-t[n-1]-t90;
					di = 0.0;
				end
				di = di + (t[n]-t[n-1]);
				if V[n+1] > V90
					# V(t[n] + t90) == V90 + o((t[n+1]-t[n])^2)
					#t90 = ((V90-V[n])/(V[n+1]-V[n]))*(t[n+1]-t[n])
					#di = di + t90
					push!(DI, di)
					di = 0.0;
				end
			end
		
		end
		
		# append storage arrays to sweep arrays
		push!(BCLs, 	BCL) 
		push!(APDs, 	APD) 
		push!(DIs, 	DI)
		push!(APAs, 	APA)
		
		# reconstruct the APD array as a delay-coordinate embedding
		try
			#R = reconstruct(APA, 1, 1)
			#ALE = numericallyapunov(R, 1:5; ntype=FixedMassNeighborhood(2))
			#ADE = takens_best_estimate(R, std(V)/4)
			ALE, ADE = compute_lyap_dim(APA)
			push!(ALEs, ALE)
			push!(ADEs, ADE)
		catch
			println("Unable to compute ALE from reconstructed APA for BCL=$(BCL).\n")
			ALE = [-Inf]	# is this a sensible default?
			ADE = [0.0]
			push!(ALEs, ALE)
			push!(ADEs, ADE)
		end
		try
			#R = reconstruct(APD, 1, 1)
			#DLE = numericallyapunov(R, 1:5; ntype=FixedMassNeighborhood(2))
			#DDE = takens_best_estimate(R, std(V)/4)
			DLE, DDE = compute_lyap_dim(APD)
			push!(DLEs, DLE)
			push!(DDEs, DDE)
		catch
			println("Unable to compute DLE from reconstructed APD for BCL=$(BCL).\n")
			DLE = [NaN]	# is this a sensible default?
			DDE = [NaN]
			push!(DLEs, DLE)
			push!(DDEs, DDE)
		end
	end

	# save data
	@save savefile APDs APAs BCLs DIs DLEs DDEs ALEs ADEs

end

# Plot (BCLs, APDs, DIs, APAs, DLEs, ALEs, DDEs, ADEs)
fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
for n=1:length(BCLs)
	m = min(length(DIs[n]), length(APDs[n]), length(APAs[n]))
	axs[1].plot(DIs[n][1:m], APDs[n][1:m], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
	axs[2].plot(DIs[n][1:m], APAs[n][1:m], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
end
axs[2].set_xlabel("DI [ms]")
axs[1].set_ylabel("APD [ms]")
axs[2].set_ylabel("APA [mV]")
tight_layout()
savefig("./figures/_BR_DI_APD_APA_$(filename).pdf")
close()

fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
for n=1:length(BCLs)
	axs[1].plot(BCLs[n]*ones(size(DDEs[n])), DDEs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=1.0)
	axs[2].plot(BCLs[n]*ones(size(ADEs[n])), ADEs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=1.0)
end
axs[1].set_ylabel("\$ d_{APD} \$")
axs[2].set_ylabel("\$ d_{APA} \$")
axs[2].set_xlabel("BCL [ms]")
#axs[2].set_xlim([0.0,400.0])
tight_layout()
savefig("./figures/_BR_BCL_DIM_$(filename).pdf")
close()

fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
for n=1:length(BCLs)
	axs[1].plot(BCLs[n]*ones(size(APAs[n])), APAs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
	axs[2].plot(BCLs[n]*ones(size(APDs[n])), APDs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
end
axs[2].set_xlabel("BCL [ms]")
axs[1].set_ylabel("APA [mV]")
axs[2].set_ylabel("APD [ms]")
#axs[2].set_xlim([0.0,400.0])
tight_layout()
savefig("./figures/_BR_BCL_APA_APD_$(filename).pdf")
close()

fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
axs[2].plot([0.0, 1000.0], [0.0, 0.0], color="red", label="")
for n=1:length(BCLs)
	axs[1].plot(BCLs[n]*ones(size(APAs[n])), APAs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
	axs[2].plot(BCLs[n]*ones(size(ALEs[n])), ALEs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.3)
end
axs[2].set_xlabel("BCL [ms]")
axs[1].set_ylabel("APA [mV]")
axs[2].set_ylabel("\$ \\lambda_1 \$ [1/ms]")
#axs[2].set_xlim([0.0,400.0])
tight_layout()
savefig("./figures/_BR_BCL_ALE_$(filename).pdf")
close()

fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
axs[2].plot([0.0, 1000.0], [0.0, 0.0], color="red", label="")
for n=1:length(BCLs)
	axs[1].plot(BCLs[n]*ones(size(APDs[n])), APDs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
	axs[2].plot(BCLs[n]*ones(size(DLEs[n])), DLEs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.3)
end
axs[2].set_xlabel("BCL [ms]")
axs[1].set_ylabel("APD [ms]")
axs[2].set_ylabel("\$ \\lambda_1 \$ [1/ms]")
#axs[2].set_xlim([0.0,400.0])
tight_layout()
savefig("./figures/_BR_BCL_DLE_$(filename).pdf")
close()
