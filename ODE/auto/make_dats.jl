using DifferentialEquations
using DelimitedFiles
using LinearAlgebra: norm
using FileIO
using JLD2

# append the path for the BR ODE model definition
push!(LOAD_PATH,"/home/chris/Development/Julia/BR/ODE/")

# use BR ODE model definition
using BR

print("Input arguments are: $(ARGS)\n");
print("\tn=$(ARGS[1])\n");
print("\tA=$(ARGS[2])\n");
print("\tf=$(ARGS[3])\n");
print("\tp=$(ARGS[4])\n");

# initial condition
u0 = BR.u0

# get the n-cycle integer
ncycle = parse(Int, ARGS[1]);

# parameters
p = BR.p
p[2] = parse(Float64, ARGS[2]); 
p[3] = parse(Float64, ARGS[3]); 
p[4] = parse(Float64, ARGS[4]);

# BCL
function BCLfromp(p)

	f = p[3];
	if p[4] > 1 && mod(p[4],2) == 0
		f = 2.0*f;
	end
	BCL = 1000.0/f;
	return BCL
end
BCL = BCLfromp(p)

# get runprob from BR
prob = BR.prob
runprob = BR.runprob

# solve system for long time
sol = runprob(prob, u0, p, (0.0, 100000.0))

# find local minimum
tt = (sol.t[end]/2.0):1.0:(sol.t[end]-ncycle*BCL)
d = zeros(length(tt))
for n = 1:length(tt)
	d[n] = norm(sol(tt[n]).-sol(tt[n]+ncycle*BCL));
end
n = sortperm(d)
q=1
while sol(tt[n[q]])[1] > -80.0 || sol(tt[n[q]])[9] > 0.0
	global q=q+1;
end
t0 = tt[n[q]]

sol = runprob(prob, sol(t0), p, (0.0, ncycle*BCL))

using PyPlot;
fig,axs = plt.subplots(1,2,figsize=(6,3), sharey=true)
axs[1].plot(sol.t[:], sol[1,:])
axs[2].plot(sol[3,:], sol[1,:])
tight_layout()
savefig("./sol.svg")
plt.close()

open("BR_sol_$(p[2])_$(p[3])_$(p[4]).dat", "w") do io
	writedlm(io, [sol.t[:] collect(transpose(sol[:,:]))])
end


