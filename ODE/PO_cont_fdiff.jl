using OrdinaryDiffEq: Tsit5, ODEProblem, remake, solve
using DynamicalSystems
using LinearAlgebra
using JLD2, FileIO
using ForwardDiff
using NLsolve
using PyPlot

push!(LOAD_PATH,pwd())
using BR

plt.style.use("seaborn-paper")

const atol=1e-09
const rtol=1e-09

# initial condition
u0 = BR.u0

# parameters
p = [1.0, 2.3, 1.0, 1.0]

# define V90 threshold
V90 = BR.V90

# get runprob from BR
prob = BR.prob

# new tspan
tspan = [0.0, 20000.0]

function BCLfromp(p::Array)

	f = p[3];
	if p[4] > 1 && mod(p[4],2) == 0
		f = 2.0*f;
	end
	BCL = 1000.0/f;
	return BCL
end

BCL = BCLfromp(p)

function resolvePO(x; prob=prob, p=p, BCL=BCL, showtrace=true)

	function G(x)
		r = similar(x)
		M = Int(length(x)/10)
		for m=1:M	
			tmp_prob = remake(prob, u0=x[(m-1)*10 .+ (1:10)], p=p, tspan=(0.0, BCLfromp(p)))
			sol = solve(tmp_prob, Tsit5(), saveat=BCL, abstol=atol, reltol=rtol, maxiters=Int(1e6))
			r[(m-1)*10 .+ (1:10)] .= sol[:,end]
		end
		r .= circshift(r,(10)) .- x
	end

	function G!(F,x)
		F.= G(x)
	end

	res = nlsolve(G!, x; ftol=1e-8, xtol=rtol*norm(x,Inf), autodiff=:forward, show_trace=showtrace)
	
	return res
	
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

# PO structure type
struct PO
	prob
	APD
	APA
	DI
	BCL
	J
end

# POs accumulation array
POs = PO[]

function accumulatePOs!(POs, x; prob=prob, p=p, BCL=BCL, V90=V90)
	
	res = resolvePO(x; prob=prob, p=p, BCL=BCL)
	
	if res.f_converged || res.residual_norm < 5e-12
		function F(x; retsol=false)
			tmp_prob = remake(prob, u0=x[1:10], p=p, tspan=(0.0, (length(x)/10)*BCL))
			sol = solve(tmp_prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=Int(1e6))
			if retsol
				return sol
			else
				return convert(Array,sol[:,end])
			end
		end
		sol = F(res.zero; retsol=true)
		APD, DI, APA = decompose_solution(sol; V90=V90)
		J = ForwardDiff.jacobian(F, res.zero)
		
		po = PO(remake(prob, u0=res.zero[1:10], p=p, tspan=(0.0, (length(res.zero)/10)*BCL)),
			APD, APA, DI, BCL,
			J)
		push!(POs, po)
		
		return true
	else
		return false
	end
end

function plotPOs(POs)

	# color / stability mapping function
	function mapping(PO)
		
		# color map first
		if isapprox(PO.prob.tspan[2]-PO.prob.tspan[1], PO.BCL; atol=0.1*PO.BCL)
			col = "black"
		elseif isapprox(PO.prob.tspan[2]-PO.prob.tspan[1], 2.0*PO.BCL; atol=2.0*0.1*PO.BCL)
			col = "C0"
		elseif isapprox(PO.prob.tspan[2]-PO.prob.tspan[1], 4.0*PO.BCL; atol=4.0*0.1*PO.BCL)
			col = "C1"
		end
		
		# stability mapping
		Λ = eigvals(PO.J)
		if maximum(abs.(Λ)) > 1.00 + atol
			ms = 3.0
		else
			ms = 5.0
		end
		
		return (col, ms)
	end

	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)

	for n=1:length(POs)
		c, ms = mapping(POs[n])
		axs[1].plot(POs[n].BCL*ones(size(POs[n].APA)), POs[n].APA, ls="none", marker=".", markersize=ms, color=c, label="")
		axs[2].plot(POs[n].BCL*ones(size(POs[n].APD)), POs[n].APD, ls="none", marker=".", markersize=ms, color=c, label="")
	end
	axs[2].set_xlabel("BCL [ms]")
	axs[1].set_ylabel("APA [mV]")
	axs[2].set_ylabel("APD [ms]")
	#axs[2].set_xlim([0.0,400.0])
	tight_layout()
	plt.savefig("./figures/PO_cont_fdiff_BCL_APD_APA_$(p[2])_$(p[4]).pdf")
	plt.close()
	
end

#=
	1:1
=#	
f=1.25
prob = remake(prob, u0=u0, p=[1.0,2.3,f,1.0], tspan=(0.0, 20000.0))
sol = solve(prob, Tsit5(), saveat=BCL)
x = sol[:,end]			# 1-cycle
prob = remake(prob, u0=x, p=[1.0,2.3,f,1.0], tspan=(0.0,BCLfromp([1.0,2.3,f,1.0])))
df= 0.0
while f <= 25.0
	success = accumulatePOs!(POs, x; prob=prob, p=[1.0,2.3,f+df,1.0], BCL=BCLfromp([1.0,2.3,f+df,1.0]), V90=V90)
	print("($(f), $(df), $(success))\n");
	if success 
		global f = max(min(f+df,25.0),1.0)
		global df = min(max(1.10*df,0.02), 1.0)
	else
		global df = max(min(0.95*df,0.02), 1.0)
	end
	local sol = solve(POs[end].prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=Int(1e6))
	global x = sol[:,1][:]

	plotPOs(POs);
end
f=1.25
prob = remake(prob, u0=u0, p=[1.0,2.3,f,1.0], tspan=(0.0, 20000.0))
sol = solve(prob, Tsit5(), saveat=BCL)
x = sol[:,end]			# 1-cycle		# 1-cycle
prob = remake(prob, u0=x, p=[1.0,2.3,f,1.0], tspan=(0.0,BCLfromp([1.0,2.3,f,1.0])))
df= 0.00
while f >= 1.0
	success = accumulatePOs!(POs, x; prob=prob, p=[1.0,2.3,f-df,1.0], BCL=BCLfromp([1.0,2.3,f-df,1.0]), V90=V90)
	print("($(f), $(df), $(success))\n");
	if success 
		global f = max(min(f+df,25.0),1.0)
		global df = min(max(1.10*df,0.02), 1.0)
	else
		global df = max(min(0.95*df,0.02), 1.0)
	end
	local sol = solve(POs[end].prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=Int(1e6))
	global x = sol[:,1][:]

	plotPOs(POs);
end

#=
	2:2
=#
f=3.125
prob = remake(prob, u0=u0, p=[1.0,2.3,f,1.0], tspan=(0.0, 20000.0))
sol = solve(prob, Tsit5(), saveat=BCLfromp([1.0,2.3,f,1.0]))
x = sol[:,(end-1):end][:]	# 2-cycle
prob = remake(prob, u0=x, p=[1.0,2.3,f,1.0], tspan=(0.0,2BCLfromp([1.0,2.3,f,1.0])))
df=0.0
# continuation
while f <= 25.0
	success = accumulatePOs!(POs, x; prob=prob, p=[1.0,2.3,f+df,1.0], BCL=BCLfromp([1.0,2.3,f+df,1.0]), V90=V90)
	print("($(f), $(df), $(success))\n");
	if success 
		global f = max(min(f+df,25.0),1.0)
		global df = min(max(1.10*df,0.02), 1.0)
	else
		global df = max(min(0.95*df,0.02), 1.0)
	end
	local sol = solve(POs[end].prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=Int(1e6))
	global x = sol[:,1:2][:]

	plotPOs(POs);
end
f=3.125
prob = remake(prob, u0=u0, p=[1.0,2.3,f,1.0], tspan=(0.0, 20000.0))
sol = solve(prob, Tsit5(), saveat=BCLfromp([1.0,2.3,f,1.0]))
x = sol[:,(end-1):end][:]	# 2-cycle
prob = remake(prob, u0=x, p=[1.0,2.3,f,1.0], tspan=(0.0,2BCLfromp([1.0,2.3,f,1.0])))
df=0.0
# continuation
while f <= 25.0
	success = accumulatePOs!(POs, x; prob=prob, p=[1.0,2.3,f-df,1.0], BCL=BCLfromp([1.0,2.3,f-df,1.0]), V90=V90)
	print("($(f), $(df), $(success))\n");
	if success 
		global f = max(min(f-df,25.0),1.0)
		global df = min(max(1.10*df,0.02), 1.0)
	else
		global df = max(min(0.95*df,0.02), 1.0)
	end
	local sol = solve(POs[end].prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=Int(1e6))
	global x = sol[:,1:2][:]

	plotPOs(POs);
end
#=
	4:4
=#
f=4.95
prob = remake(prob, u0=u0, p=[1.0,2.3,f,1.0], tspan=(0.0, 20000.0))
sol = solve(prob, Tsit5(), saveat=BCLfromp([1.0,2.3,f,1.0]))
x = sol[:,(end-3):end][:]	# 4-cycle

# continuation
df= 0.0
while f <= 25.0
	success = accumulatePOs!(POs, x; prob=prob, p=[1.0,2.3,f+df,1.0], BCL=BCLfromp([1.0,2.3,f+df,1.0]), V90=V90)
	print("($(f), $(df), $(success))\n");
	if success 
		global f = f + df
		global df = min(max(1.10*df,0.02), 1.0)
	else
		global df = max(min(0.95*df,0.02), 1.0)
	end
	local sol = solve(POs[end].prob, Tsit5(), abstol=atol, reltol=rtol, maxiters=Int(1e6))
	global x = sol[:,1:4][:]
	
	plotPOs(POs);
end
