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
p[2] =  50.0
p[4] = 500.0

accum_fig = plt.figure(figsize=(4,3))

BCL = 1000.0/6.0
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

#=
# plot sol 
BRPDE.pltsol(sol,p;preprendname="$(BCL)_BCL")

N = BRPDE.N

fig,axs = plt.subplots(2, 1, figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 3]), constrained_layout=true)

varnames=["V","Ca","X","m","h","j","d","f"]
nv=2; # for Ca-var
axs[1].plot(sol.t[:], sol[(nv-1)*N + Int(4*N/5),:], label=varnames[nv])


for nv=3:8
	axs[2].plot(sol.t[:], sol[(nv-1)*N + Int(4*N/5),:], label=varnames[nv])
end
axs[1].legend(loc="best", edgecolor="none")
axs[2].legend(loc="best", edgecolor="none")
axs[2].set_xlabel("\$ t \$ [ms]")
savefig("$(BCL)_BCL_gatings.pdf")
close()

nv=2; t_inds = Int(round(size(sol,2)/(5000.0/BCL))):1:size(sol,2)
plt.gcf()
plt.plot(sol[(nv-1)*N + Int(4*N/5),t_inds], sol[Int(4*N/5),t_inds], label="$(BCL) ms")
plt.legend(loc="best", edgecolor="none")
plt.ylabel("\$ V(t) \$")
plt.xlabel("\$ $(varnames[nv])(t) \$")
tight_layout()
plt.savefig("BCL_$(varnames[nv])V.pdf")

end

plt.close()
=#

N = BRPDE.N
x = collect(0.0:0.02:((N-1)*0.02))

plt.style.use("seaborn-paper")

fig,axs = plt.subplots(3, 1, figsize=(4,3), sharex=true, gridspec_kw=Dict("height_ratios"=> [1, 3, 1]), constrained_layout=true)

print("Ṽ ∈ [$(minimum(sol[Int(4*N/5),:])),$(maximum(sol[Int(4*N/5),:]))]\n")

axs[1].plot(sol.t, sol[Int(4*N/5),:], label="\$ \\tilde{V}(t) \$")
axs[1].set_ylabel("\$ \\tilde{V}(t) \$\n [mV]")
axs[1].set_yticks([V90,0.0])
#axs[1].set_xlabel("\$ t \$ [ms]")

pcm = axs[2].pcolormesh(sol.t, x, sol[1:N,:], edgecolor="none", shading="gouraud", rasterized=true, snap=true)
colorbar(pcm, label="\$ V(t,x) \$", orientation="vertical", ax=axs[2])
axs[2].set_ylabel("\$ x \$ [cm]")
#axs[2].set_xlabel("\$ t \$ [ms]")

axs[3].plot(sol.t, BRPDE.stimulationcurrent(sol[8N+1,:],p), label="\$ I(t) \$")
axs[3].set_ylabel("\$ I(t) \$\n[\$\\mu\$A/cm\$^2\$]")
yl = axs[3].get_ylim()
axs[3].set_yticks([-p[2], 0.0, p[2]])
axs[3].set_ylim(yl)
axs[3].set_xlabel("\$ t \$ [ms]")



savefig("./figures/BCL_$(BCL)_sol.pdf", dpi=300)
close()
