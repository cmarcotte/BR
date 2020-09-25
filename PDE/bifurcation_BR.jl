using DifferentialEquations
using DynamicalSystems
using Statistics
using Printf
using PyPlot
using JLD2, FileIO

# bifurcation_run.jl and BRPDE.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing birfurcation_run.jl and BRPDE.jl>
#	>> julia bifurcation_run.jl
push!(LOAD_PATH,pwd())
using BRPDE
#BRPDE.test()

plt.style.use("seaborn-paper")

# initial condition
u0 = BRPDE.u0

# parameters
p = BRPDE.p
p[2] = 100.0; p[4] = 501.0

# define V90 threshold
V90 = BRPDE.V90

# get runprob from BRPDE
prob = BRPDE.prob
runprob = BRPDE.runprob

# new tspan
tspan = [0.0, 20000.0]

# BCL sweep range
BCL_range = 400.0:-2.0:40.0

# savefile
savefile = "./data/data_$(p[2])_$(p[4]).jld2"

function loaddata(savefile)
	
	APDs = load(savefile,"APDs")
	APAs = load(savefile,"APAs")
	DIs  = load(savefile,"DIs" )
	BCLs = load(savefile,"BCLs")
	ALEs = load(savefile,"ALEs")
	DLEs = load(savefile,"DLEs")
	DDEs = load(savefile,"DDEs")
	ADEs = load(savefile,"ADEs")

	return (APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs)

end

function sweep(BCL_range, savefile, prob, u0, p, tspan, V90)
	
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
		if isapprox(BCL,1000.0) || isapprox(BCL,400.0) || isapprox(BCL,130.0)
			BRPDE.pltsol(sol,p;preprendname="./figures/")
	
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
		V = sol[Int(4*BRPDE.N/5),:]
		
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
			R = reconstruct(APA, 1, 1)
			ALE = numericallyapunov(R, 1:5; ntype=FixedMassNeighborhood(2))
			ADE = takens_best_estimate(R, std(V)/4)
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
			R = reconstruct(APD, 1, 1)
			DLE = numericallyapunov(R, 1:5; ntype=FixedMassNeighborhood(2))
			DDE = takens_best_estimate(R, std(V)/4)
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

	return (APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs)

end

function plotresults(APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs, p)

	# Plot (BCLs, APDs, DIs, APAs, DLEs, ALEs, DDEs, ADEs)
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for n=1:length(BCLs)
		m = min(length(DIs[n]), length(APDs[n]), length(APAs[n]))
		axs[1].plot(DIs[n][1:m], APDs[n][1:m], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
		axs[2].plot(DIs[n][1:m], APAs[n][1:m], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
	end
	axs[2].set_xlabel("DI [ms]")
	axs[1].set_ylabel("APD [ms]")
	yl = axs[1].get_ylim()
	axs[1].set_ylim([0.0,yl[2]])
	axs[2].set_ylabel("APA [mV]")
	tight_layout()
	savefig("./figures/_BR_DI_APD_APA_$(p[2])_$(p[4]).pdf")
	close()

	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for n=1:length(BCLs)
		axs[1].plot(BCLs[n]*ones(size(DDEs[n])), DDEs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=1.0)
		axs[2].plot(BCLs[n]*ones(size(ADEs[n])), ADEs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=1.0)
	end
	axs[1].set_ylabel("\$ d_{APD} \$")
	axs[2].set_ylabel("\$ d_{APA} \$")
	axs[2].set_xlabel("BCL [ms]")
	axs[2].set_xlim([0.0,maximum(collect(BCL_range))])
	tight_layout()
	savefig("./figures/_BR_BCL_DIM_$(p[2])_$(p[4]).pdf")
	close()

	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for n=1:length(BCLs)
		axs[1].plot(BCLs[n]*ones(size(APDs[n])), APDs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
		axs[2].plot(BCLs[n]*ones(size(APAs[n])), APAs[n], ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.1)
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APD [ms]")
	axs[2].set_ylabel("APA [mV]")
	axs[2].set_xlim([0.0,maximum(collect(BCL_range))])
	tight_layout()
	savefig("./figures/_BR_BCL_APA_APD_$(p[2])_$(p[4]).pdf")
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
	axs[2].set_xlim([0.0,maximum(collect(BCL_range))])
	tight_layout()
	savefig("./figures/_BR_BCL_ALE_$(p[2])_$(p[4]).pdf")
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
	axs[2].set_xlim([0.0,maximum(collect(BCL_range))])
	tight_layout()
	savefig("./figures/_BR_BCL_DLE_$(p[2])_$(p[4]).pdf")
	close()

end


try

	APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs = loaddata(savefile)
	
	print("Loaded data from $(savefile)... skipping compute.\n")
	
	# plot results
	plotresults(APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs, p)
	
catch

	print("Could not load data from $(savefile)... computing.\n")

	APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs = sweep(BCL_range, savefile, prob, u0, p, tspan, V90);

	# plot results
	plotresults(APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs, p)

end



