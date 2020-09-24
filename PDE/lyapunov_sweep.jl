using DifferentialEquations
using ODEInterfaceDiffEq
using DynamicalSystems
using Printf
using PyPlot
using JLD2, FileIO

# bifurcation_run.jl and BRPDE.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing birfurcation_run.jl and BRPDE.jl>
#	>> julia bifurcation_run.jl
push!(LOAD_PATH,pwd())
using BRPDE

plt.style.use("seaborn-paper")

# define V90 threshold
V90 = BRPDE.V90

# runmodel from model file
prob = BRPDE.prob
runprob = BRPDE.runprob

# initial condition
u0 = BRPDE.u0

N = BRPDE.N

# parameters
p = BRPDE.p
# mod p?
p[4] = 1.0

function pltandsave(BCL_range, A_range, LEs)

	plt.style.use("seaborn-paper")
	
	# Plot (BCL, A, LE)
	fig = figure(figsize = (4,3))
	
	for (m1,BCL) in enumerate(BCL_range), (m2,A) in enumerate(A_range)
		if length(LEs[m1,m2]) > 0 && length(LEs[m1,m2][1]) > 0
			if maximum(LEs[m1,m2][1]) > 0.0
				col = "tab:red"
				mss=8
				mrk="."
			else
				col = "black"
				mss=6
				mrk="."
			end
			plot(BCL , A , ls="none", marker=mrk, ms=mss, color=col)
		end
	end
	plt.xlabel("BCL [ms]")
	plt.ylabel("\$A\$ [\$\\mathrm{\\mu A/cm}^2\$]")

	tight_layout()
	savefig("_BCL_A_LE_$(p[4]).pdf")
	close()
end

function getLEs(BCL, A, LEs, m1, m2)
	# make sure the forcing frequency is passed to the model correctly
	# depending on magnitude and parity of p[4]
	f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;
	p[2] = A;

	# write out the progress
	println("Running: A=$(p[2]), f=$(p[3]), p=$(p[4]).")
	
	# new tspan
	tspan = [0.0, 20000.0]
	
	# run the model
	sol = runprob(prob, u0, p, tspan)
	
	# extract the time and voltage traces
	t = sol.t[:]
	V = sol[Int(4*N/5),:]
	
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
				apd = 0.0;
				apa = V[n];
			end
			apd = apd + (t[n]-t[n-1]);
			apa = max(apa, V[n])
			if V[n+1] < V90
				push!(APD, apd)
				push!(APA, apa)
				apd = 0.0;
				apa = V90;
			end
		end
		
		if V[n] < V90
			if V[n-1] > V90
				di = 0.0;
			end
			di = di + (t[n]-t[n-1]);
			if V[n+1] > V90
				push!(DI, di)
				di = 0.0;
			end
		end
	
	end
	
	# reconstruct the APD array as a delay-coordinate embedding
	try
		R = reconstruct(APA, 1, 1)
		LE = numericallyapunov(R, 1:5; ntype=FixedMassNeighborhood(2))
		push!(LEs[m1,m2], LE)
		pltandsave()
	catch
	end

end

function sweep(BCL_range, A_range, LEs)
	# sweep BCLs:
	for (m1,BCL) in enumerate(BCL_range), (m2,A) in enumerate(A_range)
	
		getLEs(BCL, A, LEs, m1, m2)
	
		pltandsave(BCL_range, A_range, LEs)
	end

end

# range for BCL
BCL_range = 200.0:-5.0:40.0

# range for A
A_range = 10.0:2.0:100.0

# storage
LEs = [ [] for BCL in BCL_range, A in A_range ]

sweep(BCL_range, A_range, LEs)

pltandsave(BCL_range, A_range, LEs)



