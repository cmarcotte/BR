using DifferentialEquations
using DynamicalSystems
using DelimitedFiles
using LinearAlgebra
using ForwardDiff
using Dierckx
using NLsolve
using PyPlot
using FileIO
using JLD2

push!(LOAD_PATH,pwd())
using BR

# begin by making optimized ODE / Jacobian functions and forming the archetypal ODEProblem
sys = modelingtoolkitize(BR.prob)			# take BR ODE system from existing code
ode = eval(ModelingToolkit.generate_function(sys)[2]) # ode(du, u, p, t)
jac = eval(ModelingToolkit.generate_jacobian(sys)[2]) # jac( j, u, p, t)

# this problem structure will be remake'd and reused for resolving the auto PO and for computing 
# the Jacobian of the resolved orbit
prob = ODEProblem(ODEFunction(ode, jac=jac), BR.u0, (0.0,1.0), BR.p)

# these set tolerances for the ODE solve and Newton iterations
atol = 1e-9
rtol = 1e-9
mitr = Int(1e6)

# PO structure type
struct PO
	M::Int
	u
	p
	period
	F
	J
	APD
	APA
	DI
	BCL
end

# this is for the convenience of computing the BCL from the model (forcing) parameters
function BCLfromp(p::Array)

	f = p[3];
	if p[4] > 1 && mod(p[4],2) == 0
		f = 2.0*f;
	end
	BCL = 1000.0/f;
	return BCL
end

# builds a multishooting initial guess by interpolating the auto solution
function multishooting_auto_sol(auto_sol, period, p)
	
	ind = argmin(auto_sol[:,2])
	t0 = auto_sol[ind,1]*period
	
	x = vcat(auto_sol[:,1], 	auto_sol[2:end,1].+1.0)
	y = vcat(auto_sol[:,2:end],	auto_sol[2:end,2:end])
		
	spl = ParametricSpline(x.*period, collect(transpose(y)); k=5)
	
	BCL = BCLfromp(p)
	M   = Int(round(period/BCL))
	
	u = []
	for m=0:(M-1)
		push!(u, spl(t0 + m*BCL)[:])
	end
	u = vcat(u...)
	push!(u, period)

	return (u, M)
	
end

# refines the PO from a multishooting initial guess x with params p
function resolvePO(x,p)
	
	function FJ!(F, J, x; p=p)
		# shared calculations
		M = Int((length(x)-1)/10)
		BCL = BCLfromp(p)
		function G(x)
			r = similar(x)
			r[10M+1:end] .= x[10M+1:end]
			for m=1:M	
				tmp_prob = remake(prob, u0=x[(m-1)*10 .+ (1:10)], p=p, tspan=(0.0, x[end]/M))
				sol = solve(tmp_prob, Tsit5(), saveat=BCL, abstol=atol, reltol=rtol, maxiters=mitr)
				r[(m-1)*10 .+ (1:10)] .= sol[:,end]
			end
			r[1:10M] .= circshift(r[1:10M],(10))
			r .= r .-x
			return r
		end
		
		if !(F == nothing)
			# mutating calculations specific to f! goes
			F .= G(x)
		end
		if !(J == nothing)
			# mutating calculations specific to j! goes
			J .= ForwardDiff.jacobian(G, x)
		
			dx=zeros(10)
			for m=1:M
				BR.BR!(dx, x[(m-1)*10 .+ (1:10)], p, 0.0)
				J[end,(m-1)*10 .+ (1:10)] .= dx
			end
		end
	end
	
	res = nlsolve(only_fj!(FJ!), x; ftol=atol, xtol=rtol*norm(x,Inf), show_trace=true)
	print("\n\n")
	F = similar(x)
	J = zeros(length(x), length(x))
	
	FJ!(F, J, res.zero)
	
	return (res, F, J)
end

# this computes the Floquet multipliers from the multi-shooting Jacobian (i.e., annoying indexing)
function floquet(po::PO)
	
	A = I
	A = A * (po.J[1:10, end-9:end])
	for m=1:(po.M-1)
		A = A * po.J[m*10 .+ (1:10), (m-1)*10 .+ (1:10)];
	end
	Λ = eigvals(A)
end

# this computes the APD, DI, APA from a periodic solution (t,V) with defined BCL and V90
function decompose_periodic_solution(t, V, BCL; V90=BR.V90)
	
	# storage for APD, DI, APA for this BCL
	APD = Float64[]
	APA = Float64[]
	DI  = Float64[]

	# accumulators for APD, DI, APA for this BCL
	apd = 0.0
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
	end
	
	for apd in APD
		push!(DI, ceil(BCL/apd)*BCL-apd)
	end
	
	return (APD, DI, APA)
		
end

# this function takes: an auto directory
# and iterates over the auto PO solutions in it 
# interpolates each
# computes the BCL and period from solution and parameters
# samples it at n*BCL for n=0,...,M-1
# builds a multi-shooting problem 
# resolves the PO with multishooting for more accuracy
function buildPOs(POs, basedir)
	
	bifur = readdlm("$(basedir)/_bifur");
	label = readdlm("$(basedir)/_label");
	
	fnames = readdir(basedir)
	
	for n=3:(length(fnames))
		print("\n\t Loading $(fnames[n]):\n")
		
		auto_sol = readdlm("$(basedir)/$(fnames[n])");
		period = label[n-2,12]
		p = [1.0, label[n-2,2], label[n-2,3], label[n-2,4]]
		
		u,M = multishooting_auto_sol(auto_sol, period, p)
		
		print("\tM=$(M), p=$(p), T=$(period).\n")
		
		res, F, J = resolvePO(u,p)
		
		if res.f_converged || norm(F) < atol + rtol*norm(res.zero, Inf)
			t = []; V = [];

			for m=1:M	
				tmp_prob = remake(prob, u0=u[(m-1)*10 .+ (1:10)], p=p, tspan=(u[end]*(m-1)/M, u[end]*(m)/M))
				sol = solve(tmp_prob, Tsit5(); abstol=atol, reltol=rtol, maxiters=mitr)
				push!(t, sol.t[1:end-1])
				push!(V, sol[1,1:end-1])
				if m==M	# only include endpoint of ode on final mapping
					push!(t, sol.t[end])
					push!(V, sol[1,end])
				end
			end
			
			t = vcat(t...);
			V = vcat(V...);
			
			BCL = BCLfromp(p)
			
			(APD, DI, APA) = decompose_periodic_solution(t, V, BCL)
			
			push!(POs, PO(M, u, p, period, F[1:end-1], J[1:end-1,1:end-1]+I, APD, APA, DI, BCL))
		end
	end
end

# initialize the PO list
POs = PO[]
for run in 1:5
	buildPOs(POs, "/home/chris/Development/Python/auto-07p/demos/BR/r$(run)")
end
save("./data/auto_POs.jld2", Dict("POs"=>POs));

# plot all the POs in a bifurcation diagram
#	M => color
# 	J => markersize
function plotPOs(POs)

end
