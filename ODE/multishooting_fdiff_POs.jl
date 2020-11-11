using OrdinaryDiffEq: Tsit5, ODEProblem, remake, solve
using DynamicalSystems
using LinearAlgebra
using JLD2, FileIO
using ForwardDiff
using Sundials
using NLsolve
using PyPlot

push!(LOAD_PATH,pwd())
using BR


plt.style.use("seaborn-paper")

const atol=1e-09
const rtol=1e-09
const mitr=Int(1e6)

const V90 = BR.V90

# initial condition
u0 = BR.u0

# parameters
p = BR.p

# define V90 threshold
V90 = BR.V90

# get runprob from BR
prob = BR.prob


function BCLfromp(p::Array)

	f = p[3];
	if p[4] > 1 && mod(p[4],2) == 0
		f = 2.0*f;
	end
	BCL = 1000.0/f;
	return BCL
end

# PO structure type
struct PO
	M::Int
	x
	F
	J
	APD
	APA
	DI
	BCL
end

function resolvePO(x)
	
	function FJ!(F, J, x)
		# shared calculations
		M = Int((length(x)-2)/10)
		BCL = BCLfromp([1.0,2.3,x[end-1],1.0])
		function G(x)
			r = similar(x)
			r[10M+1:end] .= x[10M+1:end]
			for m=1:M	
				tmp_prob = remake(prob, u0=x[(m-1)*10 .+ (1:10)], p=[1.0,2.3,x[end-1],1.0], tspan=(0.0, x[end]/M))
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
				BR.BR!(dx, x[(m-1)*10 .+ (1:10)], [1.0,2.3,x[end-1],1.0], 0.0)
				J[end,(m-1)*10 .+ (1:10)] .= dx
			end
			i = argmin(abs.(eigvals(J)))
			y = real.(eigvecs(J)[:,i])
			y = (I - J[end,:]*J[end,:]')*y
			J[end-1,:] .= y
		end
	end
	
	res = nlsolve(only_fj!(FJ!), x; ftol=1e-8, xtol=rtol*norm(x,Inf), show_trace=true)
	print("\n\n")
	F = similar(x)
	J = zeros(length(x), length(x))
	
	FJ!(F, J, res.zero)
	
	return (res, F, J)
end

function decompose_solution(sol; V90=V90)

	t = sol.t[:]
	V = sol[1,:]
	
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
	
	return (APD, DI, APA)
		
end

function appendPOs!(POs, x)

	res, F, J = resolvePO(x)

	tmp_prob = remake(prob, u0=res.zero[1:10], p=[1.0,2.3,x[end-1],1.0], tspan=(0.0, x[end]))
	sol = solve(tmp_prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=mitr)

	APD, DI, APA = decompose_solution(sol)	

	M = Int((length(x)-2)/10)

	if res.f_converged || norm(F) < atol + rtol*norm(res.zero, Inf)
		po = PO(M, res.zero, F[1:end-2], J[1:end-2,1:end-2]+I, APD, DI, APA, BCLfromp(tmp_prob.p))
		push!(POs, po)
	end
end

function floquet(po::PO)
	
	A = I
	A = A * (po.J[1:10, end-9:end])
	for m=1:(po.M-1)
		A = A * po.J[m*10 .+ (1:10), (m-1)*10 .+ (1:10)];
	end
	Λ = eigvals(A)
end

POs = PO[]

function trialPOs(prob, fs, ns)

	fig,axs = plt.subplots(1,2,figsize=(5.3,3.0), sharey=true)

	for (f, ncycle) in zip(fs, ns)
		print("Searching for $(ncycle)-cycle at f=$(f).\n")
		p = [1.0,2.3,f,1.0]; BCL=BCLfromp(p);
		prob = remake(prob, u0=u0, p=p, tspan=(0.0, 50BCL))
		sol = solve(prob,  CVODE_BDF(linear_solver=:GMRES), saveat=BCL, abstol=atol, reltol=rtol)
		nn = 1:size(sol, 2)-ncycle
		dd = zeros(length(nn))
		for n in nn
			dd[n] = norm(sol[:,n] .- sol[:,n+ncycle])
		end
		nx = argmin(dd)
		x = sol[:,nx-1 .+ (1:ncycle)][:]
		push!(x, f)
		push!(x, ncycle*BCL)

		appendPOs!(POs, x)
		
		prob = remake(prob, u0=POs[end].x[1:10], p=p, tspan=(0.0, POs[end].x[end]))
		sol = solve(prob, Tsit5(), abstol=atol, reltol=rtol)
		
		axs[1].plot(sol.t,	sol[1,:],	label="n=$(ncycle)")
		axs[2].plot(sol[3,:],	sol[1,:],	label="n=$(ncycle)")

		if ncycle == ns[1]
			axs[1].set_xlabel("\$ t \$")
			axs[1].set_ylabel("\$ V(t) \$")
			axs[2].set_xlabel("\$ x(t) \$")
		elseif ncycle == ns[end]
			axs[1].legend(loc=0, edgecolor="none")
		end
		print("\tΛ = $(floquet(POs[end]))\n");
	end
	tight_layout()
	plt.savefig("./figures/multishooting_fdiff_POs.pdf")
	plt.close()
end

trialPOs(prob, [1.25, 3.125, 4.95, 5.75, 6.25], [1, 2, 4, 8, 16])

