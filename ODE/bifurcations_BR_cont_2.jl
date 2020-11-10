using Bifurcations
using Bifurcations: LimitCycleProblem
using PyPlot
using OrdinaryDiffEq: Tsit5, ODEProblem, remake, solve
using Setfield: @lens
using JLD2, FileIO
using ForwardDiff, LinearAlgebra
using DynamicalSystems

plt.style.use("seaborn-paper")

const atol=1e-10
const rtol=1e-10
const mitr=Int(1e8)

struct PO
	prob
	APD
	APA
	DI
	BCL
	J
end

function ab(C,V)
	# eq (13) from original paper
	return (C[1]*exp(C[2]*(V+C[3]))+C[4]*(V+C[5]))/(exp(C[6]*(V+C[3]))+C[7])
end

function stimulationcurrent(oscillatorcomponent,p)
	return p[:A].*oscillatorcomponent.^p[:P]
end

function Casmoother(Ca; ep=1.5e-10)
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
	dx[1] = -(IK + Ix + INa + Is)/p[:C]
	dx[2] = -10^-7 * Is + 0.07*(10^-7 - x[2])
	dx[3] = ax*(1.0 - x[3]) - bx*x[3]
	dx[4] = am*(1.0 - x[4]) - bm*x[4]
	dx[5] = ah*(1.0 - x[5]) - bh*x[5]
	dx[6] = aj*(1.0 - x[6]) - bj*x[6]
	dx[7] = ad*(1.0 - x[7]) - bd*x[7]
	dx[8] = af*(1.0 - x[8]) - bf*x[8]
		
	# this is the forcing current, left boundary point only
	dx[1] = dx[1] + stimulationcurrent(x[9],p)/p[:C]

	# nonlinear oscillator dynamics
	ω = 2*pi*p[:f]/1000.0
	dx[9] = ω*x[10] + x[9]*(1.0 - x[9]^2 - x[10]^2)
	dx[10]=-ω*x[9] + x[10]*(1.0 - x[9]^2 - x[10]^2)
	
	return nothing

end

function BCLfromp(p)

	f = p[:f];
	if p[:P] > 1 && mod(p[:P],2) == 0
		f = 2.0*f;
	end
	BCL = 1000.0/f;
	return BCL
end

function decompose_solution(sol, V90=-75.0)

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
				# V(t[n-1] + t90) == V90 + o((t[n]-t[n-1])^2)
				#t90 = ((V90-V[n-1])/(V[n]-V[n-1]))*(t[n]-t[n-1])
				#apd = t90;
				apd = 0.0;
				apa = V[n];
			end
			apd = apd + (t[n]-t[n-1]);
			apa = max(apa, V[n])
			if V[n+1] < V90
				# V(t[n] + t90) == V90 + o((t[n+1]-t[n])^2)
				#t90 = ((V[n]-V90)/(V[n]-V[n+1]))*(t[n+1]-t[n])
				#apd = apd + t90
				push!(APD, apd)
				push!(APA, apa)
				apd = 0.0;
				apa = V90;
			end
		end
		
		if V[n] < V90
			if V[n-1] > V90
				#V(t[n-1]+t90) == V90 + o((t[n]-t[n-1])^2)
				#t90 = ((V[n-1]-V90)/(V[n-1]-V[n]))*(t[n]-t[n-1])
				#di = t[n]-t[n-1]-t90;
				di = 0.0;
			end
			di = di + (t[n]-t[n-1]);
			if V[n+1] > V90
				# V(t[n] + t90) == V90 + o((t[n+1]-t[n])^2)
				#t90 = ((V90-V[n])/(V[n+1]-V[n]))*(t[n+1]-t[n])
				#di = di + t90
				push!(DI, di)
				di = 0.0;
			end
		end
	
	end
	
	return (APD, DI, APA)
		
end

# PO array
POs = PO[]

# initial state
u0 = [ -84.0,10^-7,0.01,0.01,0.99,0.99,0.01,0.99,0.0,1.0]

# BCL
BCL = 1000.0

# parameters
p = (C=1.0, A=2.3, f=1000.0/BCL, P=1.0)

# base prob
prob = ODEProblem(BR!, u0, (0.0, BCL), p)

# define V90 threshold
V90 = -75.0

# build data
function build_data(LCs, POs; prob=prob)

	for n=1:length(LCs)
		
		print("Analyzing PO $(n) of $(length(LCs))...\n");
		if LCs[n].state[1,1] > -75.0
			q=argmin(abs.(LCs[n].state[1,:] .+ 80.0)) # closest state sample to -80.0 (i.e., below V90)
		else
			q=1
		end
		prob = remake(prob, u0=LCs[n].state[:,q], p=(C=1.0, A=2.3, f=LCs[n].param_value, P=1.0), tspan=(0.0, LCs[n].period))
		
		function G(x; retsol=false)
			tmp_prob = remake(prob, u0=x, p=(C=1.0, A=2.3, f=LCs[n].param_value, P=1.0), tspan=(0.0, LCs[n].period))
			sol = solve(tmp_prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=mitr)
			if !retsol
				A = convert(Array, sol)
				return A[:,end]
			else	
				return sol
			end
			
		end
		
		print("\tComputing shifted sol.\n")
		sol = G(LCs[n].state[:,q]; retsol=true)
		
		print("\t|u(T)-u(0)| = $(norm(sol(0.0).-sol(LCs[n].period)))\n");
		
		print("\tDecomposing shifted sol.\n")
		BCL = BCLfromp((C=1.0, A=2.3, f=LCs[n].param_value, P=1.0))
		APD, DI, APA = decompose_solution(sol)
		
		print("\tComputing shifted sol Jacobian.\n");
		J = ForwardDiff.jacobian(G,LCs[n].state[:,q])
		FML = eigvals(J)
		
		# if the closest eigenvalue to 1 is further away than 1e-6, show warning
		if abs(FML[argmin(abs.(FML.-(1.0+0.0im)))]-1.0+0.0im) > 1e-6
			print("eigerr = $(abs(FML[argmin(abs.(FML.-(1.0+0.0im)))]-1.0+0.0im))\n");
		end

		print("\tAccumulating PO.\n")
		po = PO(prob, APD, APA, DI, BCL, J)
		push!(POs, po)
		
	end
	
	# save data
	save("./data/bifurcations_POs_results.jld2", Dict("POs" => POs))
	
end

# make plots from data structure
function makeplots(POs)

	# color / stability mapping function
	function mapping(PO)
		
		# color map first
		if isapprox(PO.prob.tspan[2]-PO.prob.tspan[1], PO.BCL[1]; atol=0.1*PO.BCL[1])
			col = "black"
		elseif isapprox(PO.prob.tspan[2]-PO.prob.tspan[1], 2.0*PO.BCL[1]; atol=0.1*PO.BCL[1])
			col = "C0"
		elseif isapprox(PO.prob.tspan[2]-PO.prob.tspan[1], 3.0*PO.BCL[1]; atol=0.1*PO.BCL[1])
			col = "C1"
		elseif isapprox(PO.prob.tspan[2]-PO.prob.tspan[1], 4.0*PO.BCL[1]; atol=0.1*PO.BCL[1])
			col = "C2"
		else
			col = "C3"
		end
		
		# stability mapping
		if maximum(abs.(eigvals(PO.J))) > 1.00 + 1e-7
			ms = 3.0
		else
			ms = 5.0
		end
		
		return (col, ms)
	end

	Ts = zeros(length(POs));
	for n in 1:length(POs)
		Ts[n] = POs[n].prob.tspan[2]-POs[n].prob.tspan[1]
	end
	vmin = minimum(Ts)
	vmax = maximum(Ts)
	NN = sortperm(Ts; rev=true)
	
	# plot the POs
	fig,axs = plt.subplots(3,3,figsize=(8,7), sharey=true)
	sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	# fake up the array of the scalar mappable. Urgh...
	sm._A = []
	varnames = ["[Ca]","x","m","h","j","d","f","y"]
	
	for n=NN
		sol = solve(POs[n].prob, Tsit5())
		for m=2:9
			axs[m-1].plot(sol[m,:], sol[1,:], linewidth=1.0, color=plt.cm.viridis(((POs[n].prob.tspan[2]-POs[n].prob.tspan[1])-40.0)/(1000.0-40.0)))
			axs[m-1].set_xlabel("\$"*varnames[m-1]*"(t) \$")
			axs[m-1].set_ylabel("\$ V(t) \$")
		end
		axs[9].plot(sol.t./(POs[n].BCL), sol[1,:], linewidth=1.0, color=plt.cm.viridis(((POs[n].prob.tspan[2]-POs[n].prob.tspan[1])-40.0)/(1000.0-40.0)))
	end
	axs[9].set_xlabel("\$ t/\$BCL")
	axs[9].set_ylabel("\$ V(t) \$")
	for m=1:9
		clb = plt.colorbar(sm, ax=axs[m])
		clb.ax.set_title("\$ T \$")
	end
	tight_layout()
	plt.savefig("./figures/bifurcations_BR_cont_POs_$(p[:A])_$(p[:P]).pdf")
	plt.close()

	# plot the DI bifurcation diagram
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for n=NN
		c, ms = mapping(POs[n])
		m = min(length(POs[n].DI), length(POs[n].APD), length(POs[n].APA))
		axs[1].plot(POs[n].DI[1:m], POs[n].APD[1:m], ls="none", marker=".", markersize=ms, color=c, label="")
		axs[2].plot(POs[n].DI[1:m], POs[n].APA[1:m], ls="none", marker=".", markersize=ms, color=c, label="")
	end
	axs[2].set_xlabel("DI [ms]")
	axs[1].set_ylabel("APD [ms]")
	axs[2].set_ylabel("APA [mV]")
	tight_layout()
	plt.savefig("./figures/bifurcations_BR_cont_DI_APD_APA_$(p[:A])_$(p[:P]).pdf")
	plt.close()

	# plot the APD bifurcation diagram
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for n=NN
		c, ms = mapping(POs[n])
		axs[1].plot(POs[n].BCL[1]*ones(size(POs[n].APA)), POs[n].APA, ls="none", marker=".", markersize=ms, color=c, label="")
		axs[2].plot(POs[n].BCL[1]*ones(size(POs[n].APD)), POs[n].APD, ls="none", marker=".", markersize=ms, color=c, label="")
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APA [mV]")
	axs[2].set_ylabel("APD [ms]")
	#axs[2].set_xlim([0.0,400.0])
	tight_layout()
	plt.savefig("./figures/bifurcations_BR_cont_BCL_APD_APA_$(p[:A])_$(p[:P]).pdf")
	plt.close()
	
	# plot the APD bifurcation diagram
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	for n=NN
		c, ms = mapping(POs[n])
		axs[1].plot(POs[n].BCL[1]*ones(size(POs[n].APA)), POs[n].APA, ls="none", marker=".", markersize=ms, color=c, label="")
		axs[2].plot(POs[n].BCL[1]*ones(size(POs[n].APD)), POs[n].APD, ls="none", marker=".", markersize=ms, color=c, label="")
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APA [mV]")
	axs[2].set_ylabel("APD [ms]")
	#axs[2].set_xlim([0.0,400.0])
	tight_layout()
	plt.savefig("./figures/bifurcations_BR_cont_BCL_APD_APA_$(p[:A])_$(p[:P]).pdf")
	plt.close()
	
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	axs[2].plot([0.0, 1000.0], [0.0, 0.0], color="red", label="")
	for n=NN
		c, ms = mapping(POs[n])
		axs[1].plot(POs[n].BCL[1]*ones(size(POs[n].APD)), POs[n].APD, ls="none", marker=".", markersize=ms, color=c, label="")
		axs[2].plot(POs[n].BCL[1]*ones(size(POs[n].J,1)), real.(log.(Complex.(eigvals(POs[n].J)))./(POs[n].prob.tspan[2]-POs[n].prob.tspan[1])), ls="none", marker=".", markersize=ms, color=c, label="")
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APD [ms]")
	axs[2].set_ylabel("\$ \\mu \$ [1/ms]")
	#axs[2].set_xlim([0.0,400.0])
	tight_layout()
	plt.savefig("./figures/bifurcations_BR_cont_BCL_APD_FML_$(p[:A])_$(p[:P]).pdf")
	close()

	plt.close("all")

	return nothing
end

if ARGS[1] == "1"
	#= 
		1:1 response
	=#

	# initial state
	u0 = [ -84.0, 10^-7, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.0, 1.0 ]

	# parameters
	BCL = 1000.0/1.25
	p = (C=1.0, A=2.3, f=1000.0/BCL, P=1.0)

	# define V90 threshold
	V90 = -75.0

	# new tspan
	tspan = (0.0, 20000.0)

	# get prob from BR 
	prob = ODEProblem(BR!, u0, tspan, p)

	# remake prob with new u0, p, and tspan and solve
	sol = solve(prob, Tsit5())

	# get initial PO guess
	t0 = 19000.0
	u0 = sol(t0)

	prob = remake(prob, p=p, u0=u0, tspan=(0.0, BCL))

	num_mesh = 25
	degree = 5
	f_domain = (1000.0/1000.0, 1000.0/40.0)
	LCprob = LimitCycleProblem( prob, (@lens _.f), f_domain, num_mesh, degree; 
		x0=u0, l0=BCL, de_args=[Tsit5()], de_opts=(abstol=atol, reltol=rtol, maxiters=mitr) )

	solver = init(
	    LCprob;
	    start_from_nearest_root = true,
	    max_branches = 3,
	    bidirectional_first_sweep = true,
	    max_samples = 500,
	    rtol=rtol, atol=atol
	)

	# this takes a while
	#solve!(solver)
	solving!(solver) do point
		print("\npoint.i_sweep=$(point.i_sweep), point.i_point=$(point.i_point)");
		print("\n\tT = $(point.u[end-1]),\tf=$(point.u[end]),\tBCL=$(1000.0/point.u[end])\n")
	end	

	# pull the POs out of it
	LCs = Bifurcations.Codim1LimitCycle.limitcycles(solver)

	# tmp save
	save("./data/bifurcations_BR_cont_2_LCs_$(ARGS[1]).jld2", Dict("LCs"=>LCs));

elseif ARGS[1] == "2"
	#=
		2:2 response
	=#

	# initial state
	u0 = [ -84.0, 10^-7, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.0, 1.0 ]

	# parameters
	BCL = 1000.0/3.125
	p = (C=1.0, A=2.3, f=1000.0/BCL, P=1.0)

	# define V90 threshold
	V90 = -75.0

	# new tspan
	tspan = (0.0, 20000.0)

	# get prob from BR 
	prob = ODEProblem(BR!, u0, tspan, p)

	# remake prob with new u0, p, and tspan and solve
	sol = solve(prob, Tsit5())

	# get initial PO guess
	t0 = 18500.0
	u0 = sol(t0)

	prob = remake(prob, p=p, u0=u0, tspan=(0.0, 2*BCL))

	num_mesh = 25
	degree = 5
	f_domain = (1000.0/1000.0, 1000.0/40.0)
	LCprob = LimitCycleProblem( prob, (@lens _.f), f_domain, num_mesh, degree; 
		x0=u0, l0=2.0*BCL, de_args=[Tsit5()], de_opts=(abstol=atol, reltol=rtol, maxiters=mitr) )

	solver = init(
	    LCprob;
	    start_from_nearest_root = true,
	    max_branches = 3,
	    bidirectional_first_sweep = true,
	    max_samples = 500,
	    rtol=rtol, atol=atol
	)

	# this takes a while
	#solve!(solver)
	solving!(solver) do point
		print("\npoint.i_sweep=$(point.i_sweep), point.i_point=$(point.i_point)");
		print("\n\tT = $(point.u[end-1]),\tf=$(point.u[end]),\tBCL=$(1000.0/point.u[end])\n")
	end	

	# pull the POs out of it
	LCs = Bifurcations.Codim1LimitCycle.limitcycles(solver)

	# tmp save
	save("./data/bifurcations_BR_cont_2_LCs_$(ARGS[1]).jld2", Dict("LCs"=>LCs));

elseif ARGS[1] == "3"
	#=
		more 2:2	
	=#

	# initial state
	u0 = [ -84.0, 10^-7, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.0, 1.0 ]

	# parameters
	BCL = 220.0
	p = (C=1.0, A=2.3, f=1000.0/BCL, P=1.0)

	# define V90 threshold
	V90 = -75.0

	# new tspan
	tspan = (0.0, 20000.0)

	# get prob from BR 
	prob = ODEProblem(BR!, u0, tspan, p)

	# remake prob with new u0, p, and tspan and solve
	sol = solve(prob, Tsit5())

	# get initial PO guess
	t0 = 18500.0
	u0 = sol(t0)

	prob = remake(prob, p=p, u0=u0, tspan=(0.0, 2*BCL))

	num_mesh = 25
	degree = 5
	f_domain = (1000.0/1000.0, 1000.0/40.0)
	LCprob = LimitCycleProblem( prob, (@lens _.f), f_domain, num_mesh, degree; 
		x0=u0, l0=2.0*BCL, de_args=[Tsit5()], de_opts=(abstol=atol, reltol=rtol, maxiters=mitr) )

	solver = init(
	    LCprob;
	    start_from_nearest_root = true,
	    max_branches = 3,
	    bidirectional_first_sweep = true,
	    max_samples = 375,
	    rtol=rtol, atol=atol
	    
	)

	# this takes a while
	#solve!(solver)
	solving!(solver) do point
		print("\npoint.i_sweep=$(point.i_sweep), point.i_point=$(point.i_point)");
		print("\n\tT = $(point.u[end-1]),\tf=$(point.u[end]),\tBCL=$(1000.0/point.u[end])\n")
	end	

	# pull the POs out of it
	LCs = Bifurcations.Codim1LimitCycle.limitcycles(solver)

	# tmp save
	save("./data/bifurcations_BR_cont_2_LCs_$(ARGS[1]).jld2", Dict("LCs"=>LCs));

elseif ARGS[1] == "4"
	#=
		4:4 response
	=#

	# initial state
	u0 = [ -84.0, 10^-7, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.0, 1.0 ]

	# parameters
	BCL = 1000.0/4.95
	p = (C=1.0, A=2.3, f=1000.0/BCL, P=1.0)

	# define V90 threshold
	V90 = -75.0

	# new tspan
	tspan = (0.0, 100000.0)

	# get prob from BR 
	prob = ODEProblem(BR!, u0, tspan, p)

	# remake prob with new u0, p, and tspan and solve
	sol = solve(prob, Tsit5(), saveat=BCL)

	# get initial PO guess
	u0 = sol[:,end-1]

	prob = remake(prob, p=p, u0=u0, tspan=(0.0, 4*BCL))

	num_mesh = 25
	degree = 5
	f_domain = (1000.0/1000.0, 1000.0/40.0)
	LCprob = LimitCycleProblem( prob, (@lens _.f), f_domain, num_mesh, degree; 
		x0=u0, l0=4.0*BCL, de_args=[Tsit5()], de_opts=(abstol=atol, reltol=rtol, maxiters=mitr) )

	solver = init(
	    LCprob;
	    start_from_nearest_root = true,
	    max_branches = 3,
	    bidirectional_first_sweep = true,
	    max_samples = 500,
	    rtol=rtol, atol=atol
	    
	)

	# this takes a while
	#solve!(solver)
	solving!(solver) do point
		print("\npoint.i_sweep=$(point.i_sweep), point.i_point=$(point.i_point)");
		print("\n\tT = $(point.u[end-1]),\tf=$(point.u[end]),\tBCL=$(1000.0/point.u[end])\n")
	end	

	# pull the POs out of it
	LCs = Bifurcations.Codim1LimitCycle.limitcycles(solver)

	# tmp save
	save("./data/bifurcations_BR_cont_2_LCs_$(ARGS[1]).jld2", Dict("LCs"=>LCs));

elseif ARGS[1]=="5"
	#=
		3:3 response
	=#

	# initial state
	u0 = [ -84.0, 10^-7, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.0, 1.0 ]

	# parameters
	BCL = 1000.0/6.25
	p = (C=1.0, A=2.3, f=1000.0/BCL, P=1.0)

	# define V90 threshold
	V90 = -75.0

	# new tspan
	tspan = (0.0, 100000.0)

	# get prob from BR 
	prob = ODEProblem(BR!, u0, tspan, p)

	# remake prob with new u0, p, and tspan and solve
	sol = solve(prob, Tsit5())

	# get initial PO guess
	tt = 0.0:1.0:(sol.t[end]-3BCL)
	d = zeros(length(tt))
	for n=1:length(tt)
		d[n] = norm(sol(tt[n]).-sol(tt[n]+3.0*BCL))
	end
	t0 = tt[argmin(d)]
	u0 = sol(t0)

	prob = remake(prob, p=p, u0=u0, tspan=(0.0, 3*BCL))

	num_mesh = 25
	degree = 5
	f_domain = (1000.0/1000.0, 1000.0/40.0)
	LCprob = LimitCycleProblem( prob, (@lens _.f), f_domain, num_mesh, degree; 
		x0=u0, l0=3.0*BCL, de_args=[Tsit5()], de_opts=(abstol=atol, reltol=rtol, maxiters=mitr) )

	solver = init(
	    LCprob;
	    start_from_nearest_root = true,
	    max_branches = 3,
	    bidirectional_first_sweep = true,
	    max_samples = 500,
	    rtol=rtol, atol=atol
	)

	# this takes a while
	#solve!(solver)
	solving!(solver) do point
		print("\npoint.i_sweep=$(point.i_sweep), point.i_point=$(point.i_point)");
		print("\n\tT = $(point.u[end-1]),\tf=$(point.u[end]),\tBCL=$(1000.0/point.u[end])\n")
	end	

	# pull the POs out of it
	LCs = Bifurcations.Codim1LimitCycle.limitcycles(solver)

	# tmp save
	save("./data/bifurcations_BR_cont_2_LCs_$(ARGS[1]).jld2", Dict("LCs"=>LCs));
	
elseif ARGS[1] == "plot"

	POs = PO[]
	
	for n=[1 2 3]# 4]
		print("Loading './data/bifurcations_BR_cont_2_LCs_$(n).jld2'.\n");
		local LCs = load("./data/bifurcations_BR_cont_2_LCs_$(n).jld2", "LCs")
		build_data(LCs, POs);
	end
	
	# make some plots
	makeplots(POs);
	
end
