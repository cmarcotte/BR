push!(LOAD_PATH,pwd())
using BR

using OrdinaryDiffEq, ParameterizedFunctions, ODE, ODEInterface,
      ODEInterfaceDiffEq, LSODA, Sundials, DiffEqDevTools
using Plots; gr()	# using gr to plot the WorkPrecisionSet

fsprob = remake(BR.prob, u0=BR.u0, p=BR.p, tspan=(0.0,10000.0))

SOL = solve(fsprob, Vern9(), abstol=1e-14, reltol=1e-14)

test_sol = TestSolution(SOL)

abstols = 1.0 ./ 10.0 .^ (4:1:13)
reltols = 1.0 ./ 10.0 .^ (4:1:13)

#
setups =[ 
		Dict(:alg=>Tsit5())
		Dict(:alg=>radau())
		Dict(:alg=>RadauIIA5(autodiff=false))
		Dict(:alg=>CVODE_BDF(method=:Newton,linear_solver=:GMRES))
		Dict(:alg=>Rodas5(autodiff=false))
		Dict(:alg=>Kvaerno5(autodiff=false))
		Dict(:alg=>KenCarp5(autodiff=false))
		Dict(:alg=>Vern7())
		Dict(:alg=>lsoda())
	]
labels =[
		"Tsit5"
		"radau"
		"RadauIIA5"
		"CVODE_BDF"
		"Rodas5"
		"Kvaerno5"
		"KenCarp5"
		"Vern7"
		"lsoda"
	]


wp = WorkPrecisionSet(fsprob, abstols, reltols, setups; print_names=true, names=labels, appxsol=test_sol, numruns=20, error_estimate=:L2, dense_errors=true)
Plots.plot(wp)
Plots.savefig("./benchmark_ODE_wp.svg")
