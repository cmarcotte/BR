using DifferentialEquations
using ODEInterfaceDiffEq
using DynamicalSystems
using Printf
using Plots
using Roots

# bifurcation_run.jl and BR.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing birfurcation_run.jl and BR.jl>
#	>> julia bifurcation_run.jl
push!(LOAD_PATH,pwd())
using BR

# initial condition
u0 = BR.u0

# parameters
p = BR.p

# define V90 threshold
V90 = BR.V90

# runmodel from model file
runmodel = BR.runmodel

function plotsol(sol)

	plt = plot(sol, vars=(0,8), show=true)
	for n=7:-1:2
		plot!(sol, vars=(0,n), show=true)
	end
	plot!(sol.t[:], (sol.+100.0)./(140.0), vars=(0,1), linewidth=3, show=true)

end

for f in [1.00, 1.25, 1.50, 1.75, 2.00]#1000.0./(50.0:50.0:800.0)

	# write out the progress
	println("Running f=$(f).\n")

	# make sure the forcing frequency is passed to the model
	# factor of two is from the abs(sin), which is frequency-doubled
	p[3] = f
	
	for n=1:100
		# new tspan
		global tspan = [0.0, 10000.0]
		
		# run the model
		global sol = runmodel(u0, p, tspan)
		
		global u0 = sol[:,end]
	end
	
	t0 = find_zero((t) -> sol(t)[1]-V90, tspan[2]-2000.0/p[3])
	t1 = find_zero((t) -> sol(t)[1]-V90, t0 + 1000.0/p[3])
	
	plotsol(sol)
	plot!(xlim=[t0,t1])

	savefig(plt, "plt_$(f).pdf")
	
end

using SparseArrays, ForwardDiff

function F(x,p)
	dx = similar(x)
	BR.BR!(dx,x,p,0.0)
	return dx
end

J(x,p) = ForwardDiff.jacobian(x -> F(x, p), x)

A = J(BR.u0,BR.p)
eigen(A)
