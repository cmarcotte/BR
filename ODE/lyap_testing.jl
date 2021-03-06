using OrdinaryDiffEq, Sundials
using Dierckx
using Statistics
using PyPlot
using DynamicalSystems
push!(LOAD_PATH,pwd())
using BR

# initial condition
u0 = BR.u0

# params
p = BR.p; p[2] = 2.3; p[3] = 1.25; p[4] = 1.0

fig_append = "$(round(p[2],digits=2))_$(round(p[3],digits=2))_$(round(p[4],digits=2))"

# new tspan
tspan = (0.0, 40000.0)

# remake prob
prob = remake(BR.prob, u0=u0, p=p, tspan=tspan)

# sol
sol = solve(prob, CVODE_BDF(linear_solver=:GMRES); abstol=1e-13, reltol=1e-13, maxiters=Int(1e6))

function analyzeVoltageNew(t, V; V90=BR.V90)

       dt = mean(diff(t))
       Vt = Spline1D(t[:], V.-BR.V90, k=3);
       
       R = roots(Vt; maxn=Int(5e3));        # time points R: V(R) == V90
       D = derivative(Vt, R);                # V'(R)
       
       # storage for APD, DI, APA for this BCL
       APD = Float64[]
       DI  = Float64[]
       APA = Float64[]
       
       for n in 1:length(R)-1
               if D[n] > 0 && D[n+1] < 0
                       push!(APD, R[n+1]-R[n])
                       push!(APA, V90+maximum(Vt(R[n]:dt:R[n+1])))
               elseif D[n] < 0 && D[n+1] > 0
                       push!(DI, R[n+1]-R[n])
               end
       end
       return (APD, DI, APA)
end
       
function analyzeVoltage(t, V; V90=BR.V90, debug=false)

       # storage for APD, DI, APA for this BCL
       APD = Float64[]
       DI  = Float64[]
       APA = Float64[]

       # accumulators for APD, DI, APA for this BCL
       apd = 0.0
       di  = 0.0
       apa = V90

       if debug
               fig,axs = plt.subplots(4,1,figsize=(6,3),sharex=true)
               axs[1].plot(t, V, "-k")
       end
       n0 = 1; while V[n0+1] < V90; n0 = n0 + 1; end
       for n=n0:length(V)-1
               if V[n] < V90                         # if under threshold
               	apa = max(V[n],V90)
                       if V[n+1] > V90        #         increasing over threshold: Beginning of APD, end of DI
                               dt = (t[n+1]-t[n])*(V[n+1]-V90)/(V[n+1]-V[n])
                               di = di + dt
                               if n>n0
                               	push!(DI,di)
                               end
                               if debug
                               	axs[4].plot(t[n]+dt, di, ".", color="C2")
                               end
                               di = 0.0
                               apd = t[n+1]-t[n]-dt
                       elseif V[n+1] < V90        #         still under threshold: accumulate DI
                               di = di + t[n+1]-t[n]
                       end
               elseif V[n] > V90                # if over threshold
               	apa = max(apa, V[n])
                       if V[n+1] > V90        #        still over threshold: accumulate APD
                               apd = apd + t[n+1]-t[n]
                       elseif V[n+1] < V90        #        decreasing under threshold: Beginning of DI, end of APD
                               dt = (t[n+1]-t[n])*(V[n+1]-V90)/(V[n+1]-V[n])
                               apd = apd + dt
                               push!(APD,apd)
                               if debug
                               	axs[2].plot(t[n]+dt, apd, ".", color="C0")
                               end
                               apd = 0.0
                               di = t[n+1]-t[n]-dt
                               push!(APA, apa)
                               if debug
                               	axs[3].plot(t[n]+dt, apa, ".", color="C1")
                               end
                       end
               end
               if debug
               	axs[2].plot(t[n], apd, ".", color="C0")
			axs[3].plot(t[n], apa, ".", color="C1")
			axs[4].plot(t[n],  di, ".", color="C2")
			axs[1].set_xlim([t[1],t[n]])
		end
	end
	return (APD, DI, APA)
end

APD_new, DI_new, APA_new = analyzeVoltageNew(sol.t, sol[1,:]; V90=BR.V90)
APD, DI, APA = analyzeVoltage(sol.t, sol[1,:]; V90=BR.V90)

#=
fig,axs = plt.subplots(2,3,figsize=(8,4), sharex=true)
axs[1,1].plot(APD_new,		"o-", color="C0", label="cubic interp")
axs[1,2].plot(APA_new,		"o-", color="C0", label="cubic interp")
axs[1,3].plot( DI_new,		"o-", color="C0", label="cubic interp")

axs[1,1].plot(APD,		"x-", color="C1", label="linear interp")
axs[1,2].plot(APA,		"x-", color="C1", label="linear interp")
axs[1,3].plot( DI,		"x-", color="C1", label="linear interp")

axs[2,1].plot(APD_new.-APD,	"o-", color="k")
axs[2,2].plot(APA_new.-APA,	"o-", color="k")
axs[2,3].plot(DI_new.-DI,	"o-", color="k")

axs[1,1].set_title("APD [ms]")
axs[1,2].set_title("APA [mV]")
axs[1,3].set_title("DI  [ms]")

axs[1,1].legend(loc="best", edgecolor="none")
axs[1,2].legend(loc="best", edgecolor="none")
axs[1,3].legend(loc="best")

plt.tight_layout()
plt.savefig("./figures/APD_DI_APA_comparison.pdf",bbox_inches="tight")
plt.close("all")
=#

#=
Qs = 0.9:0.01:1.0
fig,axs = plt.subplots(length(Qs),1,figsize=(3,length(Qs)+1), sharex=true)
for (n,q) in enumerate(Qs)
	
	x = APD .+ q.*(APD_new.-APD)
	ks = 1:50
	lyap = []
	for (i, di) in enumerate([Euclidean(), Cityblock()])
		for m in 1:3, D in 1:10, tau in 1:10
			try
			ntype = FixedMassNeighborhood(m) #NeighborNumber(2)
			R = embed(x, D, tau)
			E = numericallyapunov(R, ks; distance = di, ntype = ntype)
			??t = 1
			?? = linear_region(ks.*??t, E)[2]
			push!(lyap,??)
			catch
			end
		end
	end
	axs[n].plot(lyap, 0.0.*lyap, ".k")
	axs[n].hist(lyap, bins="auto", density=true, alpha=0.5);
	tight_layout()
end
plt.close("all")
=#

function dolyap(x; ks=1:20, mm=1:3, DD=1:10, tt=1:10, ??t = 1)
	fig,axs = plt.subplots(2,2,figsize=(8,8), sharey="row", sharex="row")
	for (i, di) in enumerate([Euclidean(), Cityblock()])
		lyap = []
		for m in mm, D in DD, tau in tt
			try
				if D==1 && tt>1
					break;
				end
				R = embed(x, D, tau)
				E = numericallyapunov(R, ks; distance = di, ntype = FixedMassNeighborhood(m))
				?? = linear_region(ks.*??t, E)[2]
				push!(lyap,??)
				# gives the linear slope, i.e. the Lyapunov exponent
				axs[1,i].plot(ks .- 1, E .- E[1], label = "m=$m, D=$D, tau=$tau, ??=$(round(??, digits = 3))")
				global mean_l = round(mean(lyap),	digits=3)
				global min_l  = round(minimum(lyap), 	digits=3)
				global max_l  = round(maximum(lyap), 	digits=3)
				global var_l  = round(var(lyap), 	digits=3)
				axs[1,i].set_title("\$ \\lambda \\in [$min_l, $max_l] \\sim \\mathcal{N}($mean_l, $var_l)\$")
				#axs[i].legend()
				tight_layout()
			catch
			end
		end
		axs[2,i].hist(lyap, bins="auto", density=true)
		yl=axs[1,i].get_ylim()
		axs[1,i].set_xlabel("\$ k \$")
		axs[1,1].set_ylabel("\$ S(k) \$")
		axs[2,i].set_xlabel("\$ \\lambda \$")
		axs[2,1].set_ylabel("\$ P(\\lambda) \$")
		axs[2,i].set_yscale("log")
		axs[1,i].plot((ks.-1), min_l.*ks, "--r")
		axs[1,i].plot((ks.-1), max_l.*ks, "--r")
		axs[1,i].plot((ks.-1), mean_l.*ks, "--k")
		axs[1,i].set_ylim(yl)
	end
end


dolyap(APD; 		mm=1:10, DD=1:1:15, tt=1:2:75)
plt.savefig("./APD_test_$(fig_append).pdf", bbox_inches="tight")
plt.close()

dolyap(APD_new; 	mm=1:10, DD=1:1:15, tt=1:2:75)
plt.savefig("./APD_new_$(fig_append)_test.pdf", bbox_inches="tight")
plt.close()

#=
V = vcat(sol(0.0:1.0:sol.t[end]; idxs=[1])...)
dolyap(V;	ks=1:4:400, mm=1:3, DD=1:2:11, tt=1:20:400, ??t = 1.0)
plt.savefig("./V_test.pdf", bbox_inches="tight")
plt.close()
=#

fig = plt.figure()
y = abs.(APD.-APD[end])
plt.semilogy(y, "o-C0", label="APD\$_n\$ - APD\$_\\infty\$")
y = abs.(APD_new.-APD_new[end])
plt.semilogy(y, "x--C1", label="APD\$_n\$ - APD\$_\\infty\$")
inds, ll = linear_region(1:length(y), log.(y))
plt.semilogy((inds[1]:inds[end]).-1, y[inds[1]:inds[end]], ".:C2", label="linear region")
yl = plt.ylim()
plt.semilogy(1:length(y), exp.(ll.*(1:length(y))), "--k", label="\$ \\exp($(round(ll,digits=3)) k)\$")
plt.xlabel("\$ k \$")
plt.ylim(yl)
plt.legend(loc="best", edgecolor="none")
plt.savefig("./APD_$(fig_append)_truth.pdf", bbox_inches="tight")
plt.close()

fig = plt.figure()
y = abs.(APA.-APA[end])
plt.semilogy(y, "o-C0", label="APA\$_n\$ - APA\$_\\infty\$")
y = abs.(APA_new.-APA_new[end])
plt.semilogy(y, "x--C1", label="APA\$_n\$ - APA\$_\\infty\$")
inds, ll = linear_region(1:length(y), log.(y))
plt.semilogy((inds[1]:inds[end]).-1, y[inds[1]:inds[end]], ".:C2", label="linear region")
yl = plt.ylim()
plt.semilogy(1:length(y), exp.(ll.*(1:length(y))), "--k", label="\$ \\exp($(round(ll,digits=3)) k)\$")
plt.xlabel("\$ k \$")
plt.ylim(yl)
plt.legend(loc="best", edgecolor="none")
plt.savefig("./APA_$(fig_append)_truth.pdf", bbox_inches="tight")
plt.close()

fig = plt.figure()
y = abs.(DI.-DI[end])
plt.semilogy(y, "o-C0", label="DI\$_n\$ - DI\$_\\infty\$")
y = abs.(DI_new.-DI_new[end])
plt.semilogy(y, "x--C1", label="DI\$_n\$ - DI\$_\\infty\$")
inds, ll = linear_region(1:length(y), log.(y))
plt.semilogy((inds[1]:inds[end]).-1, y[inds[1]:inds[end]], ".:C2", label="linear region")
yl = plt.ylim()
plt.semilogy(1:length(y), exp.(ll.*(1:length(y))), "--k", label="\$ \\exp($(round(ll,digits=3)) k)\$")
plt.xlabel("\$ k \$")
plt.ylim(yl)
plt.legend(loc="best", edgecolor="none")
plt.savefig("./DI_$(fig_append)_truth.pdf", bbox_inches="tight")
plt.close()

