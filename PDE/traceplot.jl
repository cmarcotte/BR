using DifferentialEquations
using ODEInterfaceDiffEq
using DynamicalSystems
using Printf
using PyPlot

# bifurcation_run.jl and BRPDE.jl should be in the same directory
# and julia should be run from the same directory; e.g.,
#	>> cd <directory containing bifurcation_run.jl and BRPDE.jl>
#	>> julia bifurcation_run.jl
push!(LOAD_PATH,pwd())
using BRPDE

plt.style.use("seaborn-paper")

# initial condition
u0 = BRPDE.u0

# parameters
p = BRPDE.p
p[2] =  78.0
p[4] = 500.0

BCL =100.0
f = 1000.0/BCL; if p[4] > 1 && mod(p[4],2) == 0; f = f/2.0; end; p[3] = f;

# define V90 threshold
V90 = BRPDE.V90

# runmodel from model file
prob = BRPDE.prob
runprob = BRPDE.runprob

# collect APDs
APDs = []

# collect DIs
DIs = []

# collect periods
CLs = []

plt.style.use("seaborn-paper")
	
# new tspan
tspan = [0.0, 3000.0]

# run the model
sol = runprob(prob, u0, p, tspan)

# extract the time and voltage traces
t = sol.t[:]
V = sol[Int(4*BRPDE.N/5),:]

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

x = collect(0.0:0.02:((BRPDE.N-1)*0.02))

plt.style.use("seaborn-paper")

fig,axs = plt.subplots(4, 1, figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 2, 5, 2]), constrained_layout=true)

print("Ṽ ∈ [$(minimum(V)),$(maximum(V))]\n")

axs[1].eventplot(APD, color="black", lineoffsets=[+1],linelengths=[2])
axs[1].eventplot(DI,color="red", lineoffsets=[-1],linelengths=[2])
axs[1].set_yticks([])
axs[1].set_ylabel("AP")

axs[2].plot(t, V, label="\$ \\tilde{V}(t) \$")
axs[2].set_ylabel("\$ \\tilde{V}(t) \$\n [mV]")
axs[2].set_yticks([V90,0.0])

pcm = axs[3].pcolormesh(t, x, sol[1:BRPDE.N,:], edgecolor="none", shading="gouraud", rasterized=true, snap=true)
colorbar(pcm, label="\$ V(t,x) \$", orientation="vertical", ax=axs[3])
axs[3].set_ylabel("\$ x \$ [cm]")

axs[4].plot(t, BRPDE.stimulationcurrent(sol[8*BRPDE.N+1,:],p), label="\$ I(t) \$")
axs[4].set_ylabel("\$I(t)\$ [\$\\mathrm{\\mu A/cm}^2\$]")
yl = axs[4].get_ylim()
axs[4].set_yticks([-p[2], 0.0, p[2]])
axs[4].set_ylim(yl)
axs[4].set_xlabel("\$ t \$ [ms]")


savefig("./figures/BCL_$(BCL)_sol.pdf", dpi=300)
close()
