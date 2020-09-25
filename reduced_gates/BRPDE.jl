module BRPDE
using DifferentialEquations, Sundials
using ModelingToolkit
using PyPlot

plt.style.use("seaborn-paper")

const N = 50
const D = 0.001/(0.02*0.02)

function ab(C,V)
	# eq (13) from original paper
	return (C[1]*exp(C[2]*(V+C[3]))+C[4]*(V+C[5]))/(exp(C[6]*(V+C[3]))+C[7])
end

# the next few functions corresponds to asymptotic reduction of gates
# given:
#	q'	= a*(1-q) - b*q 
# where a=a(V) and b=b(V), then 
#        tau_q	= 1/(a+b)
#         q_oo	= a/(a+b)

function X(V)
	ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],V[n])
	bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],V[n])
	return ax/(ax+bx)
end

function m(V)
	am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],V)
	bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],V)
	return am/(am+bm)
end

function h(V)
	ah = ab([ 0.126 ,-0.25 , 77.0, 0.0, 0.0, 0.0  , 0.0],V[n])
	bh = ab([ 1.7   , 0.0  , 22.5, 0.0, 0.0,-0.082, 1.0],V[n])
	return ah/(ah+bh)
end

function j(V)
	aj = ab([ 0.055 ,-0.25 , 78.0, 0.0, 0.0,-0.2  , 1.0],V[n])
	bj = ab([ 0.3   , 0.0  , 32.0, 0.0, 0.0,-0.1  , 1.0],V[n])
	return aj/(aj+bj)
end
	
function d(V)
	ad = ab([ 0.095 ,-0.01 , -5.0, 0.0, 0.0,-0.072, 1.0],V[n])
	bd = ab([ 0.07  ,-0.017, 44.0, 0.0, 0.0, 0.05 , 1.0],V[n])
	return ad/(ad+bd)
end

function f(V)
	af = ab([ 0.012 ,-0.008, 28.0, 0.0, 0.0, 0.15 , 1.0],V[n])
	bf = ab([ 0.0065,-0.02 , 30.0, 0.0, 0.0,-0.2  , 1.0],V[n])
	return af/(af+bf)
end

function stimulationcurrent(oscillatorcomponent,p)
	return p[2].*oscillatorcomponent.^p[4]
end

function Casmoother(Ca; ep=1.5e-10)
	return ep*0.5*(1.0 + tanh(1.0-(Ca/ep))) + Ca*0.5*(1.0 + tanh((Ca/ep)-1.0))
end

function BRN!(dx,x,p,t)
	
	V = view(  x,(1+0N):(1N))
	C = view(  x,(1+1N):(2N))
	X = view(  x,(1+2N):(3N))
	#m = view(  x,(1+3N):(4N))
	h = view(  x,(1+3N):(4N))
	j = view(  x,(1+4N):(5N))
	d = view(  x,(1+5N):(6N))
	f = view(  x,(1+6N):(7N))
	
	dV = view(dx,(1+0N):(1N))
	dC = view(dx,(1+1N):(2N))
	dX = view(dx,(1+2N):(3N))
	#dm = view(dx,(1+3N):(4N))
	dh = view(dx,(1+3N):(4N))
	dj = view(dx,(1+4N):(5N))
	dd = view(dx,(1+5N):(6N))
	df = view(dx,(1+6N):(7N))
	
	# iterate over space
	@inbounds for n in 1:N
	
		# spatially local currents
		IK = (exp(0.08*(V[n]+53.0)) + exp(0.04*(V[n]+53.0)))
		IK = 4.0*(exp(0.04*(V[n]+85.0)) - 1.0)/IK
		IK = IK+0.2*(V[n]+23.0)/(1.0-exp(-0.04*(V[n]+23.0)))
		IK = 0.35*IK
		Ix = X[n]*0.8*(exp(0.04*(V[n]+77.0))-1.0)/exp(0.04*(V[n]+35.0))
		#INa= (4.0*m[n]*m[n]*m[n]*h[n]*j[n] + 0.003)*(V[n]-50.0)
		INa= (4.0*(m(V[n])^3)*h[n]*j[n] + 0.003)*(V[n]-50.0)
		Is = 0.09*d[n]*f[n]*(V[n]+82.3+13.0287*log(Casmoother(C[n])))

		# these from Beeler & Reuter table:
		ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],V[n])
		bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],V[n])
		#am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],V[n])
		#bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],V[n])
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
		#dm[n] = am*(1.0 - m[n]) - bm*m[n]
		dh[n] = ah*(1.0 - h[n]) - bh*h[n]
		dj[n] = aj*(1.0 - j[n]) - bj*j[n]
		dd[n] = ad*(1.0 - d[n]) - bd*d[n]
		df[n] = af*(1.0 - f[n]) - bf*f[n]
		
		# diffusion
		if N > 1
			if n==1
				dV[n] = dV[n] + D*(2V[n+1] - 2V[n])
			elseif n==N
				dV[n] = dV[n] + D*(2V[n-1] - 2V[n])
			else
				dV[n] = dV[n] + D*(V[n+1] + V[n-1] - 2V[n])
			end
		end
		
	end

	# this is the forcing current, left boundary point only
	dx[1] = dx[1] + stimulationcurrent(x[7N+1],p)/p[1]

	# nonlinear oscillator dynamics
	ω = 2*pi*p[3]/1000.0
	dx[7N+1] = ω*x[7N+2] + x[7N+1]*(1.0 - x[7N+1]^2 - x[7N+2]^2)
	dx[7N+2] =-ω*x[7N+1] + x[7N+2]*(1.0 - x[7N+1]^2 - x[7N+2]^2)
	
	return nothing

end

function pltCasmoother()

	Cas = -1e-6:1e-8:2e-6
	plot(Cas, [Casmoother(Ca) for Ca in Cas], linewidth=2)
	plot(Cas, [max(1.5e-7,Ca) for Ca in Cas])
	plot(Cas, 1.5e-7 .+ zeros(size(collect(Cas))), linestyle="--") 
	ylim([-1e-6,2e-6])
	savefig("./Casmoother.pdf")
	close()
end

function BRN(x,p,t)

	dx = similar(x)
	
	BRN!(dx,x,p,t)
	
	return dx
end

function runprob(prob,u0,p,tspan)

	prob = remake(prob, u0=u0, p=p, tspan=tspan)
	
	sol = solve(prob, CVODE_BDF(linear_solver=:GMRES), abstol=1e-08, reltol=1e-06, maxiters=Int(1e6))

	return sol
	
end

function pltsol(sol,p; preprendname="")

	plt.style.use("seaborn-paper")

	fig,axs = plt.subplots(2, 1, figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 3]), constrained_layout=true)

	axs[1].plot(sol.t[:], stimulationcurrent(sol[7N+1,:],p), label="\$ I(t) \$")
	for n in [1,Int(2*N/5),Int(4*N/5),N]
		axs[2].plot(sol.t[:], sol[n,:], label="\$ V(t,x=$(0.02*(n-1))) \$")
	end
	axs[2].legend(loc="best", edgecolor="none")
	axs[1].set_ylabel("\$ I(t) \$\n[\$\\mu\$A/cm\$^2\$]")
	yl = axs[1].get_ylim()
	axs[1].set_yticks([-p[2], 0.0, p[2]])
	axs[1].set_ylim(yl)
	axs[2].set_ylabel("\$ V(t,x) \$ [mV]")
	axs[2].set_xlabel("\$ t \$ [ms]")
	savefig("$(preprendname)_sol.pdf")
	close()
	
	x = collect(0.0:0.02:((N-1)*0.02))
	figure(figsize=(4,3), constrained_layout=true)
	pcolormesh(sol.t, x, sol[1:N,:], edgecolor="none", shading="gouraud", rasterized=true,snap=true)
	colorbar(label="\$ V(t,x) \$")
	ylabel("\$ x \$ [cm]")
	xlabel("\$ t \$ [ms]")
	#title("\$ V(t,x) \$")
	savefig("$(preprendname)_V(t,x).pdf",dpi=300)
	close()
	
end


# initial condition
u0 = zeros(Float64,7N+2)
u0[(0N+1):(1N)] .= -84.0
u0[(1N+1):(2N)] .= 10^-7
u0[(2N+1):(3N)] .= 0.01
#u0[(3N+1):(4N)] .= 0.01
u0[(3N+1):(4N)] .= 0.99
u0[(4N+1):(5N)] .= 0.99
u0[(5N+1):(6N)] .= 0.01
u0[(6N+1):(7N)] .= 0.99
u0[7N+1]         = 0.0
u0[7N+2]         = 1.0

# parameters
#	uF/cm2    	uA/cm2       	Hz		power
p = [	1.0,		2.3,		1.25, 		1.0]

# tspan
tspan = (0.0,300.0)

# V90 guess
V90 = -75.0

# how to make this a function which returns a prob???
print("Making optimized ODEProblem... ")
@parameters(P[1:4])
@variables(u[1:(7N+2)])

du = simplify.(BRN(u,P,0.0))
fBRN! = eval(ModelingToolkit.build_function(vec(du),vec(u),P,parallel=ModelingToolkit.MultithreadedForm())[2])

jac = ModelingToolkit.sparsejacobian(vec(du),vec(u))
fjac = eval(ModelingToolkit.build_function(jac,vec(u),P,parallel=ModelingToolkit.MultithreadedForm())[2])

if N == 1
	prob = ODEProblem(ODEFunction((du,u,p,t)->BRN!(du,u,p,t),jac = (du,u,p,t) -> fjac(du,u,p),jac_prototype = similar(jac,Float64)),u0,tspan,p)
else
	prob = ODEProblem(ODEFunction((du,u,p,t)->fBRN!(du,u,p),jac = (du,u,p,t) -> fjac(du,u,p),jac_prototype = similar(jac,Float64)),u0,tspan,p)
end

print("Done.\n")

function test()
	
	pltCasmoother()
	
	sol = runprob(BRPDE.prob,BRPDE.u0,BRPDE.p,BRPDE.tspan)
	
	pltsol(sol,p)
end

end
