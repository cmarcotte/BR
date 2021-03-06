using DifferentialEquations, Sundials
using DynamicalSystems
using Statistics
using Printf
using PyPlot
using JLD2, FileIO
using Dierckx

# bifurcation_ensemble.jl and BR.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing birfurcation_run.jl and BR.jl>
#	>> export JULIA_NUM_THREADS=[however you want to dedicate to the task]
#	>> julia bifurcation_ensemble.jl
push!(LOAD_PATH,pwd())
using BR

plt.style.use("seaborn-paper")

function freqFromBCL(BCL,p)
	f = 1000.0/BCL
	if p[4] > 1 && mod(p[4],2) == 0
		f = f/2.0
	end
	return f
end

# set up
# initial condition
u0 = BR.u0

# parameters
p = BR.p
p[2] = 2.3; p[4] = 1.0

# new tspan
tspan = (0.0, 40000.0)

# remake prob
prob = remake(BR.prob, u0=u0, p=p, tspan=tspan)

# BCL sweep range
BCL_range = 1000.0:-1.0:40.0

# saving info
filename = "$(p[2])_$(p[4])"
savefile = "./data/data_$(filename).jld2"

function compute_lyap_dim(data; ks=1:20, neigh=1:3, DD=1:7, tt=1:3, plot_stretch=false)

	lyap = []
	dim  = []
	
	if plot_stretch
		fig = figure(figsize=(10,6))
	end
	
	for (i, di) in enumerate([Euclidean(), Cityblock()])
		for n in neigh, D in DD, tau in tt
			try
				R = embed(data, D, tau)
				d = takens_best_estimate(R, std(data)/4)
				E = numericallyapunov(R, ks; distance = di, ntype=FixedMassNeighborhood(n))
				l = linear_region(ks.*1, E)[2]
				if plot_stretch
					subplot(1, 2, i)
					title("Distance: $(di)", size = 18)
					plot(ks, E, label = "D=$D, n=$n, λ=$(round(l, digits = 3))")
					if (length(neigh)*length(DD)*length(tt) < 5)
						legend(ncol=length(neigh), loc="best")#, bbox_to_anchor=(0.0, -0.25), edgecolor="none")
					end
					tight_layout()
				end
				push!(dim, d)
				push!(lyap, l)
			catch
			end
		end
	end
	return lyap, dim
end

function analyzeVoltage(t, V; V90=BR.V90)

       dt = mean(diff(t))
       Vt = Spline1D(t[:], V.-BR.V90, k=3);
       
       R = roots(Vt; maxn=Int(5e3));        # time points R: V(R) == V90
       D = derivative(Vt, R);                # V'(R)
       
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
#=
function analyzeVoltage(t,V; V90=BR.V90)

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
	
	return (APD, DI, APA)

end
=#
function output_func_analysis(sol,i; full_anal=false) # would this one work? IDK
	# print progress
	print("\n\t Analyzing solution $(n): BCL=$(BCL_range[n])\n")
	
	# extract APD, DI, APA from V(t)
	APD, DI, APA = analyzeVoltage(sol.t[:], sol[1,:])

	# construct Lyapunov exponent/dimension from APA/APD
	if isempty(APA) && !full_anal
		ALE = similar(APA)
		ADE = similar(APA)
	else
		ALE, ADE = compute_lyap_dim(APA)
	end
	if isempty(APD) && !full_anal
		DLE = similar(APD)
		DDE = similar(APD)
	else
		DLE, DDE = compute_lyap_dim(APD)
	end
	# store in appropriate indices
	return ((APD=APD, APA=APA, DI=DI, ALE=ALE, ADE=ADE, DLE=DLE, DDE=DDE), false)
end

function sweep_BCLs(prob, BCL_range)
	
	# ensemble problem generator for each BCL
	function prob_func(prob,i,repeat)
		prob.p[3] = freqFromBCL(BCL_range[i], prob.p)
		print("trajectory $(i): BCL = $(BCL_range[i]), p = $(prob.p)\n")
		return prob
	end
	ens_prob = EnsembleProblem(	prob, 
					#output_func = (sol,i) -> (sol,false),					# full sol storage
					output_func = (sol,i) -> ((t=sol.t, V=sol[1,:]), false),			# save only (t,V)
					#output_func = (sol,i) -> output_func_analysis(sol,i)			# direct (APD, APA, DI) from sol
					#output_func = (sol,i) -> output_func_analysis(sol,i; full_anal=true)	# direct (APD, APA, DI, ALE, ADE, DLE, DDE) from sol
					prob_func= prob_func)
	
	# run the ensemble problem using local threads
	sim = solve(	ens_prob, 
			CVODE_BDF(linear_solver=:GMRES), 
			EnsembleThreads(); 
			trajectories=length(BCL_range),
			batch_size=Threads.nthreads(),#8, 
			abstol=1e-14, 
			reltol=1e-14, 
			maxiters=Int(1e6)	)
	
	# collection arrays; fixed size (undef element type) for threading ease
	APDs 	= Array{Any}(undef,length(BCL_range))
	APAs 	= Array{Any}(undef,length(BCL_range))
	DIs 	= Array{Any}(undef,length(BCL_range))
	BCLs 	= Array{Any}(undef,length(BCL_range))
	ALEs 	= Array{Any}(undef,length(BCL_range))
	DLEs 	= Array{Any}(undef,length(BCL_range))
	ADEs 	= Array{Any}(undef,length(BCL_range))
	DDEs 	= Array{Any}(undef,length(BCL_range))
	
	# analyze solutions in parallel with local threading
	Threads.@threads for n in 1:length(BCL_range)

		# print progress
		print("\n\t Analyzing solution $(n): BCL=$(BCL_range[n])\n")
		
		# extract APD, DI, APA from V(t)
		APD, DI, APA = analyzeVoltage(sim[n].t, sim[n].V)
#=
		# construct Lyapunov exponent/dimension from APA/APD
		if isempty(APA)
			ALE = similar(APA)
			ADE = similar(APA)
		else
			ALE, ADE = compute_lyap_dim(APA)
		end
		if isempty(APD)
			DLE = similar(APD)
			DDE = similar(APD)
		else
			DLE, DDE = compute_lyap_dim(APD)
		end
=#		
		# store in appropriate indices
		APDs[n]	= APD
		APAs[n]	= APA
		BCLs[n]	= BCL_range[n]
		DIs[n]		= DI
#=		DLEs[n]	= DLE
		DDEs[n]	= DDE
		ALEs[n]	= ALE
		ADEs[n]	= ADE
=#		
	end
	
	return (APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs)
end

function plot_results(filename, APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs)
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
	
	return nothing
end

# main part
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

	global APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs = sweep_BCLs(prob, BCL_range);

	# save data
	save(savefile, Dict("APDs"=>APDs, "APAs"=>APAs, "BCLs"=>BCLs, "DIs"=>DIs, "DLEs"=>DLEs, "DDEs"=>DDEs, "ALEs"=>ALEs, "ADEs"=>ADEs))

end

plot_results(filename, APDs, APAs, BCLs, DIs, DLEs, DDEs, ALEs, ADEs)
