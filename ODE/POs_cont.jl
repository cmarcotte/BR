using DifferentialEquations
using DynamicalSystems
using Statistics
using LinearAlgebra
using Printf
using PyPlot
using JLD2, FileIO
using NLsolve
using Sundials

# bifurcation_run.jl and BR.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing birfurcation_run.jl and BR.jl>
#	>> julia bifurcation_run.jl
push!(LOAD_PATH,pwd())
using BR
#BR.test()

plt.style.use("seaborn-paper")

# initial condition
u0 = BR.u0

# parameters
p = BR.p
p[2] = 2.3; p[4] = 1.0

# define V90 threshold
V90 = BR.V90

# get prob from BR 
prob = BR.prob
runprob = BR.runprob
#=
	Using CVODE_BDF(linear_solver=:GMRES) seems to work for this,
	but I wonder if more accurate using less efficient, 
	but less approximate, method, like Tsit5().
	Yes, but much (100x) slower?
	Just use CVODE_BDF but with much tighter tolerances?
=#

# new tspan
tspan = [0.0, 20000.0]

# BCL sweep range
BCL_range = 1000.0:-5.0:40.0

# init
BCL = BCL_range[1]; f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;

# remake prob with new u0, p, and tspan and solve
sol = runprob(prob,u0,p,tspan)

# get initial PO guess
u0 = sol(18800.0)
tspan = (0.0, BCL)

# PO def
function F!(F,x,p,tspan)
	sol = runprob(prob, x, p, tspan)
	F .= sol[end] .- x
	return nothing
end

# J! computes Jacobian of orbit, not Jacobian of F!; if you try to use for latter, it will blow up
function J!(J,x,p,tspan; h=1e-6)
	sol = runprob(prob, x, p, tspan)
	for n=1:length(x)
		y = copy(x)
		y[n] = y[n] + h
		sol1 = runprob(prob, y, p, tspan)
		J[:,n] = sol1[end] .- sol[end]
	end
	J .= J./h
	return nothing
end

function decompose_solution(sol)

	t = sol.t[:]
	V = sol[1,:]
	
	# storage for APD, DI, APA for this BCL
	APD = Float64[]
	DI  = Float64[]
	APA = Float64[]

	# accumulators for APD, DI, APA for this BCL
	apd = 0.0
	di  = 0.0
	apa = BR.V90

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

function resolve_PO(u0,p,tspan,POs)

	res = nlsolve((F,u)->(F!(F,u,p,tspan)),u0; show_trace=true, iterations=10000) 
	if res.f_converged || res.x_converged || res.residual_norm < 1e-6 * norm(u0)
		u0 = res.zero
		
		J = zeros(Float64, length(u0), length(u0))
		J!(J, u0, p, tspan)
		Λ = eigvals(J)
		
		sol = runprob(prob, u0, p, (tspan[1],tspan[2]*2.0))

		V = sol[1,:]
		n=1; while V[n] > BR.V90; n=n+1; end;
		dsol = runprob(prob, sol[:,n], p, tspan)
		APD, DI, APA = decompose_solution(dsol)
		
		push!(POs, Dict( :u => u0, :p => p, :tspan => tspan, :Λ => Λ, :APD => APD, :DI => DI, :APA => APA ))
	end
end

function plotPOs(POs)

	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	axs[2].plot([0.0, 1000.0], [0.0, 0.0], color="red", label="")
	for PO in POs
		if maximum(abs.(PO[:Λ])) > 1.05 # some wiggle room
			ms = 3.0
		else
			ms = 5.0
		end
		axs[1].plot(ones(length(PO[:APD])).*PO[:tspan][2], PO[:APD], ls="none", marker=".", markersize=ms, color="black", label="")
		axs[2].plot(ones(length(PO[:Λ])).*PO[:tspan][2], real.(log.(Complex.(PO[:Λ]))./PO[:tspan][2]), ls="none", marker=".", markersize=3.0, color="black", label="", alpha=0.3)
	
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APD [ms]")
	axs[2].set_ylabel("\$ T^{-1} \\ln(\\Lambda) = \\lambda \$ [1/ms]")
	#axs[2].set_xlim([0.0,400.0])
	tight_layout()
	savefig("./figures/POs.pdf")
	close()
end

POs = []
resolve_PO(u0, p, tspan, POs)
plotPOs(POs)

for BCL in BCL_range[2:end]

	print("BCL=$BCL:\n")

	local f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;
	local tspan = (0.0, BCL)
	print("\tp=$p\ttspan=$tspan\n")
	resolve_PO(u0, p, tspan, POs)
	plotPOs(POs)
	
	# save data
	@save "./data/POs.jld2" POs
end

global m=1

while maximum(abs.(POs[m][:Λ])) < 1.05
	global m = m + 1
end

u0 = POs[m][:u]
p = POs[m][:p]
tspan=(0.0,20000.0)
sol = runprob(prob,u0,p,tspan)

# get initial PO guess
u0 = sol(18800.0)

# BCL range
BCL_range = (1000.0/p[3]):-5.0:40.0
BCL = 2.0*BCL_range[1]

# try to resolve the PO for the 2-cycle
resolve_PO(u0, p, (0.0, BCL), POs)
plotPOs(POs)

for BCL in BCL_range[2:end]

        print("BCL=$BCL:\n")

        local f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;
        local tspan = (0.0, 2.0*BCL)
        print("\tp=$p\ttspan=$tspan\n")
        resolve_PO(u0, p, tspan, POs)
        plotPOs(POs)

        # save data
        @save "./data/POs.jld2" POs
end
