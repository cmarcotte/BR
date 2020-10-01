using DifferentialEquations
using ODEInterfaceDiffEq
using DynamicalSystems
using Printf
using PyPlot

# bifurcation_run.jl and BR.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing bifurcation_run.jl and BR.jl>
#	>> julia bifurcation_run.jl
push!(LOAD_PATH,pwd())
using BR

plt.style.use("seaborn-paper")

# initial condition
u0 = BR.u0

# parameters
p = BR.p
p[2] = 10.0
p[4] = 500.0

BCL =1000.0/1.30
f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;

# define V90 threshold
V90 = BR.V90

# runmodel from model file
prob = BR.prob
runprob = BR.runprob

# collect APDs
APDs = []

# collect DIs
DIs = []

# collect periods
CLs = []

plt.style.use("seaborn-paper")
	
# new tspan
tspan = [0.0, 5000.0]

# run the model
sol = runprob(prob, u0, p, tspan)

# extract the time and voltage traces
t = sol.t[:]
V = sol[1,:]

# prepare to accumulate the APD and DI for this CL
APD = Float64[]
DI  = Float64[]

# look over the voltage trace, comparing to V90
for n in 2:length(t)

	if sign(V[n-1] - V90) < sign(V[n] - V90)
		push!(APD, t[n])
	elseif sign(V[n-1] - V90) > sign(V[n] - V90)
		push!(DI, t[n])
	end
	
end



plt.style.use("seaborn-paper")

fig,axs = plt.subplots(3, 1, figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 6, 2]), constrained_layout=true)

print("Ṽ ∈ [$(minimum(V)),$(maximum(V))]\n")

axs[1].eventplot(APD, color="black", lineoffsets=[+1],linelengths=[2])
axs[1].eventplot(DI,color="red", lineoffsets=[-1],linelengths=[2])
axs[1].set_yticks([])
axs[1].set_ylabel("AP")

axs[2].plot(t, V, label="\$ V(t) \$")
axs[2].set_ylabel("\$ V(t) \$\n [mV]")
axs[2].set_yticks([V90,0.0])





axs[3].plot(t, BR.stimulationcurrent(sol[9,:],p), label="\$ I(t) \$")
axs[3].set_ylabel("\$ I(t) \$\n[\$\\mu\$A/cm\$^2\$]")
yl = axs[3].get_ylim()
axs[3].set_yticks([-p[2], 0.0, p[2]])
axs[3].set_ylim(yl)
axs[3].set_xlabel("\$ t \$ [ms]")


savefig("./figures/BCL_$(BCL)_sol.pdf", dpi=300)
close()
