using DifferentialEquations
using DynamicalSystems
using Statistics
using Dierckx
using Sundials
using Printf
using PyPlot
using FileIO
using JLD2

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
p = BRPDE.p; p[2] = 100.0; p[4] = 500.0

# new tspan
tspan = (0.0, 100000.0)

# get runprob from BRPDE
prob = remake(BRPDE.prob, u0=u0, tspan=tspan, p=p)
#runprob = BRPDE.runprob

# BCL sweep range
BCL_range = 400.0:-1.0:40.0

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

function analyzeVoltage(t, V; V90=BRPDE.V90)

       dt = mean(diff(t))
       Vt = Spline1D(t[:], V.-V90, k=3);
       
       R = roots(Vt; maxn=Int(5e3));	# time points R: V(R) == V90
       D = derivative(Vt, R);		# V'(R)
       
       # storage for APD, DI, APA for this BCL
       APD = Float64[]
       DI  = Float64[]
       APA = Float64[]
       
       for n in 1:length(R)-1
               if D[n] > 0 && D[n+1] < 0
                       push!(APD, R[n+1]-R[n])
                       push!(APA, V90+maximum(Vt(R[n]:dt:R[n+1])))
               elseif D[n] < 0 && D[n+1] > 0
                       push!(DI, R[n+1]-R[n])
               end
       end
       return (APD, DI, APA)
end

function sweep(prob, BCL_range; V90=BRPDE.V90, measure_ind=Int(4*BRPDE.N/5))
	
	# callback for saving the state at t: V(t) == V90
	condition(u,t,integrator) = u[measure_ind]-V90
	affect!(integrator) = nothing
	cb = ContinuousCallback(condition,affect!,save_positions=(true,false))
	
	# collect APDs
	APDs = []

	# collect APAs
	APAs = []

	# collect DIs
	DIs = []

	# collect periods
	BCLs = []

	# sweep BCLs:
	for BCL in BCL_range

		# write out the progress
		println("Running BCL=$(BCL).\n")

		# make sure the forcing frequency is passed to the model correctly 
		# depending on magnitude and parity of p[4]
		f = 1000.0/BCL; if prob.p[4] > 1 && mod(prob.p[4],2) == 0; f = f/2.0; end; prob.p[3] = f;

#= 
	CVODE_BDF is fastest at fixed abstol/reltol. 
	The actual error is higher than other methods, and it saves the whole solution.
	CVODE_BDF doesn't handle the callback which further improves the interpolation error in time for V(t)=V90.
 	Vern7 is much slower, but yields lower actual error with higher tolerances -- making the APDs produced useful -- 
 	where CVODE_BDF, while fast, may not give useful APD sequences.
 	Additionally, Vern7 works with save_idxs to reduce the memory usage by a factor of O(6e-3).
 	This would make running several PDE solves in parallel fit in memory.
 	Alternatively, ROCK4() supports the same callback and save_idxs features as Vern7, while being especially suited
 	for parabolic systems... ROCK4() uses O(1e-3) of the memory of CVODE_BDF.
=#
#=
		# remake prob with new u0, p, and tspan and solve
		sol = solve(prob, CVODE_BDF(linear_solver=:GMRES); abstol=1e-08, reltol=1e-06, maxiters=Int(1e8))
		
		# generate APD, DI, and APA sequences from interpolated V(t,x=0.8)
		APD,DI,APA = analyzeVoltage(sol.t, sol[measure_ind,:]; V90=V90)
=#		
		# remake prob with new u0, p, and tspan and solve
		sol = solve(prob, ROCK4(); abstol=1e-06, reltol=1e-06, maxiters=Int(1e7), callback=cb, save_idxs=[measure_ind]) 
		
		# generate APD, DI, and APA sequences from interpolated V(t,x=0.8)
		APD,DI,APA = analyzeVoltage(sol.t, sol[1,:]; V90=V90)	# using save_idxs=[measure_ind] makes sol[1,:] = V(t,x=0.8)
		
		# append sequences to collection arrays
		push!(APDs, APD)
		push!( DIs,  DI)
		push!(APAs, APA)
		push!(BCLs, BCL)
	end

	# save data
	@save savefile APDs APAs BCLs DIs

	return (APDs, APAs, BCLs, DIs)

end

APDs, APAs, BCLs, DIs = sweep(prob, BCL_range)


function plotresults(APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs; fig_append="$(p[2])_$(p[4])")

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
	savefig("./figures/_BR_DI_APD_APA_$(fig_append).pdf")
	close()
#=
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
	savefig("./figures/_BR_BCL_DIM_$(fig_append).pdf")
	close()
=#
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
	savefig("./figures/_BR_BCL_APA_APD_$(fig_append).pdf")
	close()
#=
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
	savefig("./figures/_BR_BCL_ALE_$(fig_append).pdf")
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
	savefig("./figures/_BR_BCL_DLE_$(fig_append).pdf")
	close()
=#
end

#=
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
=#


