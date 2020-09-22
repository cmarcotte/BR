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
p[3] = 5.00
p[4] = 500.0

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

# plot the solution, with the APD/DI boundary points
fig,axs = plt.subplots(3,1,figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [2, 1, 5]), constrained_layout=true)

axs[1].plot(t./1000.0, BR.stimulationcurrent(sol[9,:],p))
axs[1].set_ylabel("\$ I(t) \$ \n [\$\\mu\$A/cm\$^2\$]")
yl = axs[1].get_ylim()
axs[1].set_yticks([-p[2],0.0,p[2]])
axs[1].set_yticklabels(["",0,"\$ A \$"])
axs[1].set_ylim(yl)

axs[2].eventplot(APD./1000.0, color="black", lineoffsets=[+1],linelengths=[2])
axs[2].eventplot(DI./1000.0,color="red", lineoffsets=[-1],linelengths=[2])
axs[2].set_yticks([])
axs[2].set_ylabel("AP")

axs[3].plot(t./1000.0, V)
axs[3].set_ylabel("\$ V(t) \$ [mV]")
axs[3].set_yticks([V90, 0.0])
axs[3].set_yticklabels(["\$V_{90}\$", 0])
#axs[3].plot(APD./1000.0, V90*ones(size(APD)), ls="none", marker=".", ms=5.0, color="black", label="")
#axs[3].plot( DI./1000.0, V90*ones(size( DI)), ls="none", marker=".", ms=5.0, color="red", label="")
axs[3].set_xlabel("\$ t \$ [s]")

#tight_layout()
savefig("./f_$(p[2])_$(p[3])_$(p[4]).pdf")

close()

