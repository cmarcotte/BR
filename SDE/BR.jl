module BR
using DifferentialEquations, Sundials
using PyPlot

plt.style.use("seaborn-paper")

const N = 1

function ab(C,V)
	# eq (13) from original paper
	return (C[1]*exp(C[2]*(V+C[3]))+C[4]*(V+C[5]))/(exp(C[6]*(V+C[3]))+C[7])
end

function stimulationcurrent(oscillatorcomponent,p)
	return p[2].*oscillatorcomponent.^p[4]
end

function Casmoother(Ca; ep=1e-7)
	return ep*0.5*(1.0 + tanh(1.0-(Ca/ep))) + Ca*0.5*(1.0 + tanh((Ca/ep)-1.0))
end

function BR!(dx,x,p,t)
	
	# spatially local currents
	IK = (exp(0.08*(x[1]+53.0)) + exp(0.04*(x[1]+53.0)))
	IK = 4.0*(exp(0.04*(x[1]+85.0)) - 1.0)/IK
	IK = IK+0.2*(x[1]+23.0)/(1.0-exp(-0.04*(x[1]+23.0)))
	IK = 0.35*IK
	Ix = x[3]*0.8*(exp(0.04*(x[1]+77.0))-1.0)/exp(0.04*(x[1]+35.0))
	INa= (4.0*x[4]*x[4]*x[4]*x[5]*x[6] + 0.003)*(x[1]-50.0)
	Is = 0.09*x[7]*x[8]*(x[1]+82.3+13.0287*log(Casmoother(x[2])))

	# these from Beeler & Reuter table:
	ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],x[1])
	bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],x[1])
	am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],x[1])
	bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],x[1])
	ah = ab([ 0.126 ,-0.25 , 77.0, 0.0, 0.0, 0.0  , 0.0],x[1])
	bh = ab([ 1.7   , 0.0  , 22.5, 0.0, 0.0,-0.082, 1.0],x[1])
	aj = ab([ 0.055 ,-0.25 , 78.0, 0.0, 0.0,-0.2  , 1.0],x[1])
	bj = ab([ 0.3   , 0.0  , 32.0, 0.0, 0.0,-0.1  , 1.0],x[1])
	ad = ab([ 0.095 ,-0.01 , -5.0, 0.0, 0.0,-0.072, 1.0],x[1])
	bd = ab([ 0.07  ,-0.017, 44.0, 0.0, 0.0, 0.05 , 1.0],x[1])
	af = ab([ 0.012 ,-0.008, 28.0, 0.0, 0.0, 0.15 , 1.0],x[1])
	bf = ab([ 0.0065,-0.02 , 30.0, 0.0, 0.0,-0.2  , 1.0],x[1])

	# BR dynamics
	dx[1] = -(IK + Ix + INa + Is)/p[1]
	dx[2] = -10^-7 * Is + 0.07*(10^-7 - x[2])
	dx[3] = ax*(1.0 - x[3]) - bx*x[3]
	dx[4] = am*(1.0 - x[4]) - bm*x[4]
	dx[5] = ah*(1.0 - x[5]) - bh*x[5]
	dx[6] = aj*(1.0 - x[6]) - bj*x[6]
	dx[7] = ad*(1.0 - x[7]) - bd*x[7]
	dx[8] = af*(1.0 - x[8]) - bf*x[8]
		
	# this is the forcing current, left boundary point only
	dx[1] = dx[1] + stimulationcurrent(x[8N+1],p)/p[1]

	# nonlinear oscillator dynamics
	ω = 2*pi*p[3]/1000.0
	dx[9]  = ω*x[10] + x[9]*(1.0 - x[9]^2 - x[10]^2)
	dx[10] =-ω*x[9] + x[10]*(1.0 - x[9]^2 - x[10]^2)
	
	return nothing

end

function noise!(dx, x, p, t)
	
	# current noise
	dx[1] = 1.0/p[1]
	# calcium noise
	#dx[2] = 1e-7; 
	# gatings noise
	#dx[3:8] = 1.0
	# noisy amplitude and phase of forcing oscillator
	#dx[9]  = 1.0
	#dx[10] = 1.0

	return nothing
	
end

function pltCasmoother()

	Cas = -1e-6:1e-8:2e-6
	plot(Cas, [Casmoother(Ca) for Ca in Cas], linewidth=2)
	plot(Cas, [max(1.5e-7,Ca) for Ca in Cas])
	plot(Cas, 1.5e-7 .+ zeros(size(collect(Cas))), linestyle="--") 
	ylim([1e-13,2e-6])
	yscale("log")
	savefig("./Casmoother.pdf")
	close()
end

function runprob(prob,u0,p,tspan)

	prob = remake(prob, u0=u0, p=p, tspan=tspan)
	
	sol = solve(prob, SOSRA2())
	
	return sol
	
end

function pltsol(sol,p)

	plt.style.use("seaborn-paper")

	fig,axs = plt.subplots(2, 1, figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 3]), constrained_layout=true)

	axs[1].plot(sol.t[:], stimulationcurrent(sol[8N+1,:],p), label="\$ I(t) \$")
	axs[2].plot(sol.t[:], sol[1,:], label="\$ V(t) \$")
	axs[2].legend(loc="best", edgecolor="none")
	axs[1].set_ylabel("\$ I(t) \$\n[\$\\mu\$A/cm\$^2\$]")
	axs[2].set_ylabel("\$ V(t,x) \$ [mV]")
	axs[2].set_xlabel("\$ t \$ [ms]")
	savefig("sol.pdf")
	close()
	
end


# initial condition
u0 = [ -84.0,10^-7,0.01,0.01,0.99,0.99,0.01,0.99,0.0,1.0]

# parameters
#	uF/cm2    	uA/cm2       	Hz		power
p = [	1.0,		2.3,		1.25, 		1.0]

# tspan
tspan = (0.0,300.0)

# V90 guess
V90 = -75.0

# default prob
prob = SDEProblem(BR!, noise!, u0, tspan, p)

function test()
	
	pltCasmoother()
	
	# initial condition
	u0 = [ -84.0,10^-7,0.01,0.01,0.99,0.99,0.01,0.99,0.0,1.0]

	# parameters
	#	uF/cm2    	uA/cm2       	Hz		power
	p = [	1.0,		2.3,		1.25, 		1.0]

	# tspan
	tspan = (0.0,5000.0)
	
	sol = runprob(BR.prob,u0,p,tspan)
	
	pltsol(sol,p)
end

end
