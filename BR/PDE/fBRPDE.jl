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

function localBRkinetics!(dx::SubArray, x::SubArray, p)
	"""
		dx is a view of the local state update array
		x is a view of the local state array
		p is an Array{Any,1}
		
		The idea is we update dx and x in-place using local
		information (the supplied indices of the views only)
		and this might be non-allocating?
	"""
		
	IK=(exp(0.08*(x[1]+53.0)) + exp(0.04*(x[1]+53.0)))
	IK=4.0*(exp(0.04*(x[1]+85.0)) - 1.0)/IK
	IK=IK+0.2*(x[1]+23.0)/(1.0-exp(-0.04*(x[1]+23.0)))
	IK=0.35*IK
	Ix = x[3]*0.8*(exp(0.04*(x[1]+77.0))-1.0)/exp(0.04*(x[1]+35.0))
	INa = (4.0*x[4]*x[4]*x[4]*x[5]*x[6] + 0.003)*(x[1]-50.0)
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
	
	return nothing
	
end

function BRPDE_fortran_layout!(dx,x,p,t)
	
	@inbounds for n in 1:N
	
		#localBRkinetics!(view(dx,(8*(n-1)+1):(8*(n-1)+8)), view( x,(8*(n-1)+1):(8*(n-1)+8)), p)
		
		V = x[8*(n-1)+1]
		
		IK=(exp(0.08*(V+53.0)) + exp(0.04*(V+53.0)))
		IK=4.0*(exp(0.04*(V+85.0)) - 1.0)/IK
		IK=IK+0.2*(V+23.0)/(1.0-exp(-0.04*(V+23.0)))
		IK=0.35*IK
		Ix = x[8*(n-1)+3]*0.8*(exp(0.04*(V+77.0))-1.0)/exp(0.04*(V+35.0))
		INa = (4.0*x[8*(n-1)+4]*x[8*(n-1)+4]*x[8*(n-1)+4]*x[8*(n-1)+5]*x[8*(n-1)+6] + 0.003)*(V-50.0)
		Is = 0.09*x[8*(n-1)+7]*x[8*(n-1)+8]*(V+82.3+13.0287*log(Casmoother(x[8*(n-1)+2])))
		
		# these from Beeler & Reuter table:
		ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],V)
		bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],V)
		am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],V)
		bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],V)
		ah = ab([ 0.126 ,-0.25 , 77.0, 0.0, 0.0, 0.0  , 0.0],V)
		bh = ab([ 1.7   , 0.0  , 22.5, 0.0, 0.0,-0.082, 1.0],V)
		aj = ab([ 0.055 ,-0.25 , 78.0, 0.0, 0.0,-0.2  , 1.0],V)
		bj = ab([ 0.3   , 0.0  , 32.0, 0.0, 0.0,-0.1  , 1.0],V)
		ad = ab([ 0.095 ,-0.01 , -5.0, 0.0, 0.0,-0.072, 1.0],V)
		bd = ab([ 0.07  ,-0.017, 44.0, 0.0, 0.0, 0.05 , 1.0],V)
		af = ab([ 0.012 ,-0.008, 28.0, 0.0, 0.0, 0.15 , 1.0],V)
		bf = ab([ 0.0065,-0.02 , 30.0, 0.0, 0.0,-0.2  , 1.0],V)
		
		# BR dynamics
		dx[8*(n-1)+1] = -(IK + Ix + INa + Is)/p[1]
		dx[8*(n-1)+2] = -10^-7 * Is + 0.07*(10^-7 - x[8*(n-1)+2])
		dx[8*(n-1)+3] = ax*(1.0 - x[8*(n-1)+3]) - bx*x[8*(n-1)+3]
		dx[8*(n-1)+4] = am*(1.0 - x[8*(n-1)+4]) - bm*x[8*(n-1)+4]
		dx[8*(n-1)+5] = ah*(1.0 - x[8*(n-1)+5]) - bh*x[8*(n-1)+5]
		dx[8*(n-1)+6] = aj*(1.0 - x[8*(n-1)+6]) - bj*x[8*(n-1)+6]
		dx[8*(n-1)+7] = ad*(1.0 - x[8*(n-1)+7]) - bd*x[8*(n-1)+7]
		dx[8*(n-1)+8] = af*(1.0 - x[8*(n-1)+8]) - bf*x[8*(n-1)+8]
		
		# diffusion
		if N > 1
			if n==1
				dx[8*(n-1)+1] = dx[8*(n-1)+1] + (0.001/(0.02*0.02))*(2x[8*(n+0)+1] - 2x[8*(n-1)+1])
			elseif n==N
				dx[8*(n-1)+1] = dx[8*(n-1)+1] + (0.001/(0.02*0.02))*(2x[8*(n-2)+1] - 2x[8*(n-1)+1])
			else
				dx[8*(n-1)+1] = dx[8*(n-1)+1] + (0.001/(0.02*0.02))*(x[8*(n-2)+1] + x[8*(n+0)+1] - 2x[8*(n-1)+1])
			end
		end
		
	end
	
	# this is the forcing current, left boundary point only
	dx[1] = dx[1] + stimulationcurrent(x[8N+1],p)/p[1]

	# nonlinear oscillator dynamics
	ω = 2*pi*p[3]/1000.0
	dx[8*N+1] = ω*x[8*N+2] + x[8*N+1]*(1.0 - x[8*N+1]^2 - x[8*N+2]^2)
	dx[8*N+2] =-ω*x[8*N+1] + x[8*N+2]*(1.0 - x[8*N+1]^2 - x[8*N+2]^2)
	
	return nothing

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

# how to show the spyplot of jac/Sjac?
jac_sparsity = ones(Float64,8N+2,8N+2)
for n=1:8N+2 
	for m=1:8N+2
		if jac[n,m] === ModelingToolkit.Constant(0)
			jac_sparsity[n,m] = 0.0
		end
	end
end
jac_sparsity = sparse(jac_sparsity)
spy(jac_sparsity)
xticks([0,8N+2])
#xticklabels([1,8N+2])
yticks([0,8N+2])
#yticklabels([1,8N+2])
nnz = length(nonzeros(jac_sparsity));
nzp = round(nnz*100.0/(jac_sparsity.m * jac_sparsity.n),digits=2)
title("nnz = $(nnz), $(nzp)%")
tight_layout()
savefig("./jac_sparsity_N_$(N).pdf")
close()

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
	p = [	1.0,		2.3,		3.0, 		1.0]
end

CL = 1000.0/p[3]; if p[4]>1 && mod(p[4],2)==0; CL = CL/2.0; end;

# time span
tspan = (0.0,1000.0)

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
		                           jac_prototype = similar(Sjac,Float64)), #jac_prototype = jac_sparsity), #jac_prototype = similar(Sjac,Float64)),
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
                                   jac_prototype = similar(Sjac,Float64)),
                                   x,tspan,p)

if N==1
	prob = ssprob
else
	prob = fsprob
end

const atol=1e-08;
const rtol=1e-06;
println("N=$(N), abstol=$(atol), reltol=$(rtol), tspan=($(tspan[1]),$(tspan[2])).")
println("Benchmarking f, F, j, J:")                          
@btime BRPDE!(dx,x,p,0.0)         
@btime fBRPDE!(dx,x,p)                                  
@btime djac(similar(jac,Float64),x,p)
@btime sjac(similar(Sjac,Float64),x,p)

ts = collect(tspan[1]:0.5:tspan[2])
SOL = solve(fsprob, Vern9(), abstol=1e-14, reltol=1e-14)

# reuse truth sol
sol = SOL;

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
savefig("f_sol.pdf")
close()

if N>1
	x = collect(0.0:0.02:((N-1)*0.02))
	figure(figsize=(4,3), constrained_layout=true)
	pcolormesh(sol.t, x, sol[1:N,:], edgecolor="none", shading="gouraud", rasterized=true)
	colorbar(label="\$ V(t,x) \$")
	ylabel("\$ x \$ [cm]")
	xlabel("\$ t \$ [ms]")
	#title("\$ V(t,x) \$")
	savefig("./f_V(t,x).pdf",dpi=300)
	close()
end

function errsol(sol)

	weights = ones(Float64,8N+2)
	
	#weights[(0N+1):(1N)] .= 1.0/mean(sol[(0N+1):(1N),:][:])
	#weights[(1N+1):(2N)] .= 1.0/mean(sol[(1N+1):(2N),:][:])
	
	errs = collect(transpose(sol(sol.t[:]) .- SOL(sol.t[:]))) # length(ts) x 8N+2
	
	return sqrt.(sum((errs*weights).^2,dims=2))

end

solver = "CVODE_BDF";
println("Solver = $(solver).")
@btime solve(prob, CVODE_BDF(linear_solver=:GMRES), abstol=atol, reltol=rtol);
sol =  solve(prob, CVODE_BDF(linear_solver=:GMRES), abstol=atol, reltol=rtol);
plot(sol.t[:], errsol(sol), linestyle="-", alpha=0.8, label="$(solver).", color="C4")

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
savefig("CVODE_BDF_sol.pdf")
close()

errs = sol(sol.t[:]) .- SOL(sol.t[:]) # 8N+2 x length(ts)

fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 3]), constrained_layout=true)

axs[1].plot(sol.t[:], errs[8N+1,:], label="\$ \\delta I(t) \$")
n=1; axs[2].plot(sol.t[:], errs[n,:], label="\$ \\delta V(t,x=$(0.02*(n-1))) \$")
if N>1
        for n in 1:5
                axs[2].plot(sol.t[:], errs[Int(n*N/5),:], label="\$ \\delta V(t,x=$(0.02*Int(n*N/5))) \$")
        end
end
axs[2].legend(loc="best", edgecolor="none")
axs[1].set_ylabel("\$ \\delta I(t) \$\n[\$\\mu\$A/cm\$^2\$]")
axs[2].set_ylabel("\$ \\delta V(t,x) \$ [mV]")
axs[2].set_xlabel("\$ t \$ [ms]")
savefig("CVODE_BDF_sol_err.pdf")
close()


if N>1
        x = collect(0.0:0.02:((N-1)*0.02))
        figure(figsize=(4,3), constrained_layout=true)
        pcolormesh(sol.t, x, abs.(errs[1:N,:]), edgecolor="none", shading="gouraud", rasterized=true)
        colorbar(label="\$ \\delta V(t,x) \$")
        ylabel("\$ x \$ [cm]")
        xlabel("\$ t \$ [ms]")
        #title("\$ V(t,x) \$")
        savefig("./e_V(t,x).pdf",dpi=300)
        close()
end


solver = "Tsit5";
println("Solver = $(solver).")
@btime solve(prob, Tsit5(), abstol=atol, reltol=rtol);
sol =  solve(prob, Tsit5(), abstol=atol, reltol=rtol);
plot(sol.t[:], errsol(sol), linestyle="-", alpha=0.8, label="$(solver).", color="C2")

solver = "ROCK4";
println("Solver = $(solver).")
@btime solve(prob, ROCK4(), abstol=atol, reltol=rtol);
sol =  solve(prob, ROCK4(), abstol=atol, reltol=rtol);
plot(sol.t[:], errsol(sol), linestyle="-", alpha=0.8, label="$(solver).", color="C3")

solver = "Rodas5";
println("Solver = $(solver).")
@btime solve(prob, Rodas5(autodiff=false), abstol=atol, reltol=rtol);
sol =  solve(prob, Rodas5(autodiff=false), abstol=atol, reltol=rtol);
plot(sol.t[:], errsol(sol), linestyle="-", alpha=0.8, label="$(solver).", color="C5")

solver = "Rodas4P";
println("Solver = $(solver).")
@btime solve(prob, Rodas4P(autodiff=false), abstol=atol, reltol=rtol);
sol =  solve(prob, Rodas4P(autodiff=false), abstol=atol, reltol=rtol);
plot(sol.t[:], errsol(sol), linestyle="-", alpha=0.8, label="$(solver).", color="C6")

solver = "KenCarp5";
println("Solver = $(solver).")
@btime solve(prob, KenCarp5(), abstol=atol, reltol=rtol);
sol =  solve(prob, KenCarp5(), abstol=atol, reltol=rtol);
plot(sol.t[:], errsol(sol), linestyle="-", alpha=0.8, label="$(solver).", color="C8")

xlabel("\$ t \$ [ms]")
ylabel("\$ \\sqrt{\\sum_{i}^{8N+2} (\\mathbf{u}_i(t) - \\mathbf{U}_i(t))^2} \$")
yscale("log")
ylim([min(atol,rtol)*1e-2, 1e2])
gcf().set_size_inches((4,3))
legend(loc="best", ncol=2, edgecolor="none")
tight_layout()
savefig("./fastcomp.pdf")
close()



