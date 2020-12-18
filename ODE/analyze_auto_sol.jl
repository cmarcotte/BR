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

# for plotting style
plt.style.use("seaborn-paper")

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
	print("\t |u - u_0| = $(norm(res.zero .- x))\n")
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
	
	# t[1] == 0.0
	# t[end] == period
	# V[1] == V[end]
	# find index q for t & V so that V[q] <= V90 && V[q] > V90
	# then make t[q] == 0.0
	lt=length(t)
	q=1; while q < lt-1 && ~(V[q] <= V90 && V[q+1] > V90); q=q+1; end; #print("Index q=$(q), length(t)=$(lt).\n");
	t = vcat(t, t[end].+t[2:end])
	V = vcat(V, V[2:end])
	
	# storage for APD, DI, APA for this BCL
	APD = Float64[]
	APA = Float64[]
	DI  = Float64[]

	# accumulators for APD, DI, APA for this BCL
	apd = 0.0
	apa = V90
	di  = 0.0

	# look over the voltage trace, comparing to V90
	for n in q:(q+lt-1)
	
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
	
	if length(DI) == 1+length(APD)
		di = popat!(DI,1)
		DI[end] = DI[end] + di
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
				tmp_prob = remake(prob, u0=res.zero[(m-1)*10 .+ (1:10)], p=p, tspan=(res.zero[end]*(m-1)/M, res.zero[end]*(m)/M))
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
			
			push!(POs, PO(M, res.zero[1:end-1], p, res.zero[end], F[1:end-1], J[1:end-1,1:end-1]+I, APD, APA, DI, BCL))
		end
	end
end

# plot all the POs in a bifurcation diagram
#	M => color
# 	J => markersize
function plotPOs(POs)

	function plotparams(po::PO)

		Λ = floquet(po)
		stable = maximum(abs.(Λ)) > 1.0+atol+rtol*norm(po.u,Inf) ? false : true
		
		if stable
			ms = 6.0
		else
			ms = 2.0
		end
		
		if po.M == 1
			cl = "black"
		else
			cl = "C$(Int(log2(po.M)-1))"
		end
		
		al = 1.0 #sqrt(1.0/po.M)
		
		return (ms, cl, al)
	end

	

	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for m = [1,2,4,8,16]
		for po in POs
			if po.M == m
				(ms,cl,al) = plotparams(po)
			
				axs[2].plot((1000.0/po.BCL)*ones(length(po.APD)), po.APD, ".", markersize=ms, color=cl, alpha=al)
				axs[1].plot((1000.0/po.BCL)*ones(length(po.APA)), po.APA, ".", markersize=ms, color=cl, alpha=al)
			end
		end
	end
	axs[2].set_xlabel("\$ f \$ [Hz]")
	axs[1].set_ylabel("APA [mV]")
	axs[2].set_ylabel("APD [ms]")
	plt.savefig("./figures/auto_POs_f.pdf", bbox_inches="tight", dpi=300)
	plt.close()	

	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for m=[1,2,4,8,16]
		for po in POs
			if po.M == m
				(ms,cl,al) = plotparams(po)
			
				axs[2].plot(po.BCL*ones(length(po.APD)), po.APD, ".", markersize=ms, color=cl, alpha=al)
				axs[1].plot(po.BCL*ones(length(po.APA)), po.APA, ".", markersize=ms, color=cl, alpha=al)
			end
		end
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APA [mV]")
	axs[2].set_ylabel("APD [ms]")
	plt.savefig("./figures/auto_POs.pdf", bbox_inches="tight", dpi=300)
	plt.close()
	
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for m=[1,2,4,8,16]
		for po in POs
			if po.M == m
				(ms,cl,al) = plotparams(po)
			
				axs[2].plot(po.DI, po.APD, ".", markersize=ms, color=cl, alpha=al)
				axs[1].plot(po.DI, po.APA, ".", markersize=ms, color=cl, alpha=al)
			end
		end
	end
	axs[2].set_xlabel("DI [ms]")
	axs[1].set_ylabel("APA [mV]")
	axs[2].set_ylabel("APD [ms]")
	plt.savefig("./figures/auto_POs_DI.pdf", bbox_inches="tight", dpi=300)
	plt.close()
	
	
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for m=[1,2,4,8,16]
		for po in POs
			if po.M == m
				(ms,cl,al) = plotparams(po)
				Λ = floquet(po)
				axs[1].plot(po.BCL*ones(length(po.APD)), po.APD, ".", markersize=ms, color=cl, alpha=al)
				axs[2].plot(po.BCL*ones(length(Λ)), real.(log.(Complex.(Λ))./po.period), ".", markersize=ms, color=cl, alpha=al)
			end
		end
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APD [ms]")
	axs[2].set_ylabel("\$ \\mu\$ [ms]\$^{-1}\$")
	plt.savefig("./figures/auto_POs_floquet.pdf", bbox_inches="tight", dpi=300)
	plt.close()
	
	fig = plt.figure(figsize=(4,3))
	for m=[1,2,4,8,16]
		for po in POs
			if po.M == m
				(ms,cl,al) = plotparams(po)
				
				for n=1:po.M
					plt.plot(circshift(po.APD,1-n), circshift(po.APD,-n), ".", markersize=ms, color=cl, alpha=al)
				end
			end
		end
	end
	plt.xlabel("APD\$_{n}\$ [ms]")
	plt.ylabel("APD\$_{n+1}\$ [ms]")
	plt.savefig("./figures/auto_POs_APD_map.pdf", bbox_inches="tight", dpi=300)
	plt.close()
	
	fig = plt.figure(figsize=(4,3))
	for m=[1,2,4,8,16]
		for po in POs
			if po.M == m
				(ms,cl,al) = plotparams(po)
			
				for n=1:po.M
					plt.plot(circshift(po.APA,1-n), circshift(po.APA,-n), ".", markersize=ms, color=cl, alpha=al)
				end
			end
		end
	end
	plt.xlabel("APA\$_{n}\$ [ms]")
	plt.ylabel("APA\$_{n+1}\$ [ms]")
	plt.savefig("./figures/auto_POs_APA_map.pdf", bbox_inches="tight", dpi=300)
	plt.close()
	
end



# initialize the PO list
POs = PO[]
posload = ARGS[1] == "true" ? true : false;
if ~posload
	print("\nComputing POs...\n\n");
	for run in 1:7
		buildPOs(POs, "/home/chris/Development/Python/auto-07p/demos/BR/r$(run)")
		plotPOs(POs)
	end
	save("./data/auto_POs.jld2", Dict("POs"=>POs));
else
	print("\nLoading POs...\n\n");
	global POs = load("./data/auto_POs.jld2", "POs");
end

plotPOs(POs)

# plot sample of each branch as (t,V) & (x,V) for some fixed f
f = 6.25; p = [1.0, 2.3, f, 1.0]; BCL = BCLfromp(p);
fig,axs = plt.subplots(1,2,figsize=(4,2), sharey=true)
# find the indices for each branch
for M in [1,2,4,8,16]
	inds = []
	for (i,po) in enumerate(POs)
		if po.M == M
			push!(inds, i)
		end
	end
	# find the PO on the branch closest to the prescribed BCL
	ind = argmin(abs.([POs[i].BCL for i in inds] .- BCL))
	# resolve the closest po
	res, F, J = resolvePO(vcat(POs[inds[ind]].u,M*BCL), p)
	
	# if it converges to a po
	if res.f_converged || norm(F) < atol + rtol*norm(res.zero, Inf)
		t = []; Sol = [];

		for m=1:M	
			tmp_prob = remake(prob, u0=res.zero[(m-1)*10 .+ (1:10)], p=p, tspan=(res.zero[end]*(m-1)/M, res.zero[end]*(m)/M))
			sol = solve(tmp_prob, Tsit5(); abstol=atol, reltol=rtol, maxiters=mitr)
			for n in 1:length(sol.t)-1
				push!(t, 	sol.t[n])
				push!(Sol, 	sol[:,n])
			end
			if m==M	# only include endpoint of ode on final mapping
				push!(t,  sol.t[end])
				push!(Sol,sol[:,end])
			end
		end
		
		t = vcat(t...);
		Sol = vcat(Sol...); Sol = reshape(Sol, (10, Int(length(Sol)/10)));
		V = Sol[1,:][:];
		
		if M == 1
			cl = "black"
		else
			cl = "C$(Int(log2(M)-1))"
		end
		
		axs[1].plot(t, 	Sol[1,:], "-", color=cl, linewidth=sqrt(1.0/M), label="\$ n=$(M)\$")
		axs[2].plot(Sol[3,:], 	Sol[1,:], "-", color=cl, linewidth=sqrt(1.0/M))
	end
	axs[1].legend(loc=0, edgecolor="none")
	axs[1].set_xlabel("\$ t \$ [ms]")
	axs[1].set_ylabel("\$ V(t) \$ [mV]")
	axs[2].set_xlabel("\$ x(t) \$")
	fig.suptitle("\$ f = $(f)\$ [Hz]")
	plt.savefig("./figures/auto_po_loops_f_$(f).pdf", bbox_inches="tight", dpi=300)	
end
plt.close()

