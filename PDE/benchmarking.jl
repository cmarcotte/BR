using DifferentialEquations
using ODEInterface, ODEInterfaceDiffEq, Sundials
using ModelingToolkit
using BenchmarkTools, LinearAlgebra, SparseArrays
using SparsityDetection
using PyPlot
using Printf

plt.style.use("seaborn-paper")

const N = 50

function ab(C,V)
	# eq (13) from original paper
	return (C[1]*exp(C[2]*(V+C[3]))+C[4]*(V+C[5]))/(exp(C[6]*(V+C[3]))+C[7])
end

function stimulationcurrent(oscillatorcomponent,p)
	return p[2].*oscillatorcomponent.^p[4]
end

function Casmoother(Ca; ep=1.5e-7)
	#return ep + Ca*log(1.0 + exp(Ca/ep))
	return ep*0.5*(1.0 + tanh(1.0-(Ca/ep))) + Ca*0.5*(1.0 + tanh((Ca/ep)-1.0))
end

function BRPDE!(dx,x,p,t)
	
	V = view(  x,(1+0N):(1N))
	C = view(  x,(1+1N):(2N))
	X = view(  x,(1+2N):(3N))
	m = view(  x,(1+3N):(4N))
	h = view(  x,(1+4N):(5N))
	j = view(  x,(1+5N):(6N))
	d = view(  x,(1+6N):(7N))
	f = view(  x,(1+7N):(8N))
	
	dV = view(dx,(1+0N):(1N))
	dC = view(dx,(1+1N):(2N))
	dX = view(dx,(1+2N):(3N))
	dm = view(dx,(1+3N):(4N))
	dh = view(dx,(1+4N):(5N))
	dj = view(dx,(1+5N):(6N))
	dd = view(dx,(1+6N):(7N))
	df = view(dx,(1+7N):(8N))
	
	# iterate over space
	@inbounds for n in 1:N
	
		# spatially local currents
		IK = (exp(0.08*(V[n]+53.0)) + exp(0.04*(V[n]+53.0)))
		IK = 4.0*(exp(0.04*(V[n]+85.0)) - 1.0)/IK
		IK = IK+0.2*(V[n]+23.0)/(1.0-exp(-0.04*(V[n]+23.0)))
		IK = 0.35*IK
		Ix = X[n]*0.8*(exp(0.04*(V[n]+77.0))-1.0)/exp(0.04*(V[n]+35.0))
		INa= (4.0*m[n]*m[n]*m[n]*h[n]*j[n] + 0.003)*(V[n]-50.0)
		Is = 0.09*d[n]*f[n]*(V[n]+82.3+13.0287*log(Casmoother(C[n])))

		# these from Beeler & Reuter table:
		ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],V[n])
		bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],V[n])
		am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],V[n])
		bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],V[n])
		ah = ab([ 0.126 ,-0.25 , 77.0, 0.0, 0.0, 0.0  , 0.0],V[n])
		bh = ab([ 1.7   , 0.0  , 22.5, 0.0, 0.0,-0.082, 1.0],V[n])
		aj = ab([ 0.055 ,-0.25 , 78.0, 0.0, 0.0,-0.2  , 1.0],V[n])
		bj = ab([ 0.3   , 0.0  , 32.0, 0.0, 0.0,-0.1  , 1.0],V[n])
		ad = ab([ 0.095 ,-0.01 , -5.0, 0.0, 0.0,-0.072, 1.0],V[n])
		bd = ab([ 0.07  ,-0.017, 44.0, 0.0, 0.0, 0.05 , 1.0],V[n])
		af = ab([ 0.012 ,-0.008, 28.0, 0.0, 0.0, 0.15 , 1.0],V[n])
		bf = ab([ 0.0065,-0.02 , 30.0, 0.0, 0.0,-0.2  , 1.0],V[n])

		# BR dynamics
		dV[n] = -(IK + Ix + INa + Is)/p[1]
		dC[n] = -10^-7 * Is + 0.07*(10^-7 - C[n])
		dX[n] = ax*(1.0 - X[n]) - bx*X[n]
		dm[n] = am*(1.0 - m[n]) - bm*m[n]
		dh[n] = ah*(1.0 - h[n]) - bh*h[n]
		dj[n] = aj*(1.0 - j[n]) - bj*j[n]
		dd[n] = ad*(1.0 - d[n]) - bd*d[n]
		df[n] = af*(1.0 - f[n]) - bf*f[n]
		
		# diffusion
		if N > 1
			if n==1
				dV[n] = dV[n] + (0.001/(0.02*0.02))*(2V[n+1] - 2V[n])
			elseif n==N
				dV[n] = dV[n] + (0.001/(0.02*0.02))*(2V[n-1] - 2V[n])
			else
				dV[n] = dV[n] + (0.001/(0.02*0.02))*(V[n+1] + V[n-1] - 2V[n])
			end
		end
		
	end
	
	# this is the forcing current, left boundary point only
	dx[1] = dx[1] + stimulationcurrent(x[8N+1],p)/p[1]

	# nonlinear oscillator dynamics
	ω = 2*pi*p[3]/1000.0
	dx[8N+1] = ω*x[8N+2] + x[8N+1]*(1.0 - x[8N+1]^2 - x[8N+2]^2)
	dx[8N+2] =-ω*x[8N+1] + x[8N+2]*(1.0 - x[8N+1]^2 - x[8N+2]^2)
	
	return nothing

end

function BRPDE(x,p,t)

	dx = similar(x)
	
	BRPDE!(dx,x,p,t)
	
	return dx
end

@parameters P[1:4]
@variables u[1:(8N+2)]

du = simplify.(BRPDE(u,P,0.0))
fBRPDE! = eval(ModelingToolkit.build_function(vec(du),vec(u),P,
            parallel=ModelingToolkit.MultithreadedForm())[2])

jac = simplify.(ModelingToolkit.jacobian(vec(du),vec(u)))
djac = eval(ModelingToolkit.build_function(jac,vec(u),P,
            parallel=ModelingToolkit.MultithreadedForm())[2])
Sjac = simplify(ModelingToolkit.sparsejacobian(vec(du),vec(u)))
sjac = eval(ModelingToolkit.build_function(Sjac,vec(u),P,
            parallel=ModelingToolkit.MultithreadedForm())[2])

# state
x = zeros(Float64,8N+2)
x[(0N+1):(1N)] .= -84.0
x[(1N+1):(2N)] .= 10^-7
x[(2N+1):(3N)] .= 0.01
x[(3N+1):(4N)] .= 0.01
x[(4N+1):(5N)] .= 0.99
x[(5N+1):(6N)] .= 0.99
x[(6N+1):(7N)] .= 0.01
x[(7N+1):(8N)] .= 0.99
x[8N+1]         = 0.0
x[8N+2]         = 1.0

# update
dx = zeros(Float64,8N+2)

# parameters
if N>1
	#	uF/cm2    	uA/cm2       	Hz		power
	p = [	1.0,		50.0,		3.0, 		500.0]

else
	p = [	1.0,		10.0,		3.0, 		500.0]
end

CL = 1000.0/p[3]; if p[4]>1 && mod(p[4],2)==0; CL = CL/2.0; end;

# time span
tspan = (0.0, 1000.0)

# build problems:
# 	slow f, no jac:
snprob = ODEProblem(BRPDE!,x,tspan,p)
# 	slow f, dense jac:
sdprob = ODEProblem(ODEFunction((du,u,p,t)->BRPDE!(du,u,p,t),
		                           jac = (du,u,p,t) -> djac(du,u,p),
		                           jac_prototype = similar(jac,Float64)),
		                           x,tspan,p)
# 	slow f, sparse jac:
ssprob = ODEProblem(ODEFunction((du,u,p,t)->BRPDE!(du,u,p,t),
		                           jac = (du,u,p,t) -> sjac(du,u,p),
		                           jac_prototype = similar(Sjac,Float64)), #jac_prototype = jac_sparsity), 
		                           x,tspan,p)
#	fast f, no jac:
fnprob = ODEProblem(ODEFunction((du,u,p,t)->fBRPDE!(du,u,p)),x,tspan,p)
# 	fast f, dense jac:
fdprob = ODEProblem(ODEFunction((du,u,p,t)->fBRPDE!(du,u,p),
		                           jac = (du,u,p,t) -> djac(du,u,p),
		                           jac_prototype = similar(jac,Float64)),
		                           x,tspan,p)
# 	fast f, sparse jac:
fsprob = ODEProblem(ODEFunction((du,u,p,t)->fBRPDE!(du,u,p),
                                   jac = (du,u,p,t) -> sjac(du,u,p),
                                   jac_prototype = similar(Sjac,Float64)),#jac_prototype = jac_sparsity), 
                                   x,tspan,p)

# plot truth sol
function plottruth(sol)
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 3]), constrained_layout=true)

	axs[1].plot(sol.t[:], stimulationcurrent(sol[8N+1,:],p), label="\$ I(t) \$")
	n=1; axs[2].plot(sol.t[:], sol[n,:], label="\$ V(t,x=$(0.02*(n-1))) \$")
	if N>1
		for n in 1:5
			axs[2].plot(sol.t[:], sol[Int(n*N/5),:], label="\$ V(t,x=$(0.02*Int(n*N/5))) \$")
		end
	end
	axs[2].legend(loc="best", edgecolor="none")
	axs[1].set_ylabel("\$ I(t) \$\n[\$\\mu\$A/cm\$^2\$]")
	axs[2].set_ylabel("\$ V(t,x) \$ [mV]")
	axs[2].set_xlabel("\$ t \$ [ms]")
	plt.savefig("b_sol.pdf")
	plt.close()

	if N>1
		x = collect(0.0:0.02:((N-1)*0.02))
		plt.figure(figsize=(4,3), constrained_layout=true)
		plt.pcolormesh(sol.t, x, sol[1:N,:], edgecolor="none", shading="gouraud", rasterized=true)
		plt.colorbar(label="\$ V(t,x) \$")
		plt.ylabel("\$ x \$ [cm]")
		plt.xlabel("\$ t \$ [ms]")
		#title("\$ V(t,x) \$")
		plt.savefig("./b_V(t,x).pdf",dpi=300)
		plt.close()
	end

end

# solve for truth with tight tolerances and a high-order method with a very good interpolant;
# can't beat Vern9() for this
SOL = solve(fsprob, Vern9(), abstol=1e-14, reltol=1e-14)
plottruth(SOL)

using OrdinaryDiffEq, ParameterizedFunctions, ODE, ODEInterface,
      ODEInterfaceDiffEq, LSODA, Sundials, DiffEqDevTools
using Plots; pyplot()	# using gr to plot the WorkPrecisionSet
    
abstols = 1.0 ./ 10.0 .^ (6:0.5:12)
reltols = 1.0 ./ 10.0 .^ (4:0.5:10)

test_sol = TestSolution(SOL)

#
setups =[ 
		Dict(:alg=>Tsit5())
		Dict(:alg=>ROCK4())
		Dict(:alg=>CVODE_BDF(method=:Newton,linear_solver=:GMRES))
		Dict(:alg=>Rodas5(autodiff=false))
		Dict(:alg=>Kvaerno5(autodiff=false))
		Dict(:alg=>KenCarp5(autodiff=false))
		Dict(:alg=>Vern7())
		Dict(:alg=>Vern8())
	]
labels =[
		"Tsit5"
		"ROCK4"
		"CVODE_BDF(:Newton,:GMRES)"
		"Rodas5"
		"Kvaerno5"
		"KenCarp5"
		"Vern7"
		"Vern8"
	]


wp = WorkPrecisionSet(snprob, abstols, reltols, setups; print_names=true, names=labels, appxsol=test_sol, numruns=20, error_estimate=:L2, dense_errors=true)
Plots.plot(wp)
Plots.savefig("./N_$(N)_snprob_wp.svg")

wp = WorkPrecisionSet(sdprob, abstols, reltols, setups; print_names=true, names=labels, appxsol=test_sol, numruns=20, error_estimate=:L2, dense_errors=true)
Plots.plot(wp)
Plots.savefig("./N_$(N)_sdprob_wp.svg")

wp = WorkPrecisionSet(ssprob, abstols, reltols, setups; print_names=true, names=labels, appxsol=test_sol, numruns=20, error_estimate=:L2, dense_errors=true)
Plots.plot(wp)
Plots.savefig("./N_$(N)_ssprob_wp.svg")

wp = WorkPrecisionSet(fnprob, abstols, reltols, setups; print_names=true, names=labels, appxsol=test_sol, numruns=20, error_estimate=:L2, dense_errors=true)
Plots.plot(wp)
Plots.savefig("./N_$(N)_fnprob_wp.svg")

wp = WorkPrecisionSet(fdprob, abstols, reltols, setups; print_names=true, names=labels, appxsol=test_sol, numruns=20, error_estimate=:L2, dense_errors=true)
Plots.plot(wp)
Plots.savefig("./N_$(N)_fdprob_wp.svg")

wp = WorkPrecisionSet(fsprob, abstols, reltols, setups; print_names=true, names=labels, appxsol=test_sol, numruns=20, error_estimate=:L2, dense_errors=true)
Plots.plot(wp)
Plots.savefig("./N_$(N)_fsprob_wp.svg")
