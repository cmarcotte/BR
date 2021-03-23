using DifferentialEquations, ModelingToolkit

const N = 1
const D = 0.001/(0.02*0.02)

function ab(C,V)
	# eq (13) from original paper
	return (C[1]*exp(C[2]*(V+C[3]))+C[4]*(V+C[5]))/(exp(C[6]*(V+C[3]))+C[7])
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
				dV[n] = dV[n] + D*(2V[n+1] - 2V[n])
			elseif n==N
				dV[n] = dV[n] + D*(2V[n-1] - 2V[n])
			else
				dV[n] = dV[n] + D*(V[n+1] + V[n-1] - 2V[n])
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

# initial condition
u0 = zeros(Float64,8N+2)
u0[(0N+1):(1N)] .= -84.0
u0[(1N+1):(2N)] .= 10^-7
u0[(2N+1):(3N)] .= 0.01
u0[(3N+1):(4N)] .= 0.01
u0[(4N+1):(5N)] .= 0.99
u0[(5N+1):(6N)] .= 0.99
u0[(6N+1):(7N)] .= 0.01
u0[(7N+1):(8N)] .= 0.99
u0[8N+1]         = 0.0
u0[8N+2]         = 1.0

# parameters
#	uF/cm2    	uA/cm2       	Hz		power
p = [	1.0,		2.3,		1.25, 		1.0]

# tspan
tspan = (0.0,5000.0)


prob	= ODEProblem(BRN!, u0, tspan, p)
sys	= modelingtoolkitize(prob)

f!	= eval(ModelingToolkit.generate_function(sys,parallel=ModelingToolkit.MultithreadedForm())[2])
J!	= eval(ModelingToolkit.generate_jacobian(sys,sparse=true,parallel=ModelingToolkit.MultithreadedForm())[2])

prob	= ODEProblem(ODEFunction(f!, jac=J!), u0, tspan, p)
	
