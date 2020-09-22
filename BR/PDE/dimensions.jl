using DifferentialEquations
using DynamicalSystems
using Statistics
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
p = BRPDE.p
p[2] = 50.0; p[4] = 500.0

# define V90 threshold
V90 = BRPDE.V90

# get runprob from BRPDE
prob = BRPDE.prob
runprob = BRPDE.runprob

# get N from BRPDE
const N = BRPDE.N

# new tspan
tspan = [0.0, 20000.0]

# BCL sweep range
BCL_range = 1000.0:-10.0:40.0

# figure
fig,axs = plt.subplots(4,1,figsize=(4,4), sharex=true)

# sweep BCLs:
for BCL in BCL_range

	# determine forcing oscillator frequency
	f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;

	# write out the progress
	print("Running BCL=$(BCL).\n")
	
	# remake prob with new u0, p, and tspan and solve
	sol = runprob(prob,u0,p,tspan)
	
	# estimate dimension from V(t,x) for a few different x's
	for n in 1:5
		V = sol[Int(n*N/5),:]
		R = reconstruct(V, 1, 1)
		
		try
			DIM = generalized_dim(0, R)
			axs[1].plot(BCL, DIM, linestyle="none", marker=".", markersize=5, color="C$(n-1)", alpha=0.8)
		catch
		end
		try	
			DIM = generalized_dim(1, R)
			axs[2].plot(BCL, DIM, linestyle="none", marker=".", markersize=5, color="C$(n-1)", alpha=0.8)
		catch
		end
		try	
			DIM = generalized_dim(2, R)
			axs[3].plot(BCL, DIM, linestyle="none", marker=".", markersize=5, color="C$(n-1)", alpha=0.8)
		catch
		end
		try	
			DIM = takens_best_estimate(R, std(V)/4)
			axs[4].plot(BCL, DIM, linestyle="none", marker=".", markersize=5, color="C$(n-1)", alpha=0.8)
		catch
		end
	end
	
	axs[1].set_ylabel("\$ d_0 \$")
	axs[2].set_ylabel("\$ d_1 \$")
	axs[3].set_ylabel("\$ d_2 \$")
	axs[4].set_ylabel("\$ d_T \$")
	axs[4].set_xlabel(" BCL [ms] ")
	tight_layout()
	savefig("./dims.pdf")
end
close()

