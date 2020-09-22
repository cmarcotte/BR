push!(LOAD_PATH,pwd())
using map

using DynamicalSystems

# form dynamical system
ds = DiscreteDynamicalSystem(map.f, map.u0, map.p0)

# orbit diagram making
u_index = 1
p_index = 1
pvalues = 400.0:-1.0:1.0
n = 1000
Ttr = 0

output = orbitdiagram(ds, u_index, p_index, pvalues; n = n, Ttr = Ttr, dt = 1)

# data manipulation for plotting
L = length(pvalues)
x = Vector{Float64}(undef, n*L)
y = copy(x)
for j in 1:L
    x[(1 + (j-1)*n):j*n] .= pvalues[j]
    y[(1 + (j-1)*n):j*n] .= output[j]
end

# plotting
using PyPlot
figure(figsize=(4,3))
plot(x, y, ls = "None", ms = 1.0, color = "black", marker = ".", alpha = 0.01)
xlim(0,400); ylim(0,300)
xlabel("\$c\$"); ylabel("\$a\$")
tight_layout()
savefig("./bifdiag.pdf", bbox_inches="tight", pad_inches=0.0, dpi=300)
close()

# lyapunov exponents
lya = []
for c in pvalues
	p = [c, map.p0[2]]
	
		ds = DiscreteDynamicalSystem(map.f, 300.0*rand(1), p)
		
		# does not work for some reason?
		#l = lyapunovs(ds, 1000, k=1, Ttr = 0, dt = 1)
		
		# manual:
		tr1 = trajectory(ds, 100)
		u2 = get_state(ds) + (1e-9 * ones(dimension(ds)))
		tr2 = trajectory(ds, 100, u2)
		
		using LinearAlgebra: norm
	
		fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
		
		# Plot the x-coordinate of the two trajectories:
		axs[1].plot(tr1[:, 1], alpha = 0.5)
		axs[1].plot(tr2[:, 1], alpha = 0.5)
		axs[1].ylabel("a")
		
		# Plot their distance in a semilog plot:
		d = [norm(tr1[i] - tr2[i]) for i in 1:length(tr2)]
		axs[2].ylabel("da"); 
		axs[2].xlabel("n");
		axs[2].semilogy(d, alpha = 0.5);
		
		# subset consisting of exponential growth/decay
		i=1
		q = i
		while d[i] > 1e-15 && d[i] < 1e-1 && i < length(d)
			q = i
			i = i + 1
		end
		
		using LsqFit
		@. model(x,y) = y[1] + y[2]*x
		fit = curve_fit(model, (1:q), log.(d[1:q]), [d[1],0.0])
		axs[2].semilogy(1:q, exp.(model(1:q,coef(fit))), alpha=0.5, label="fit")
		
		tight_layout()
		savefig("./mapdiag.pdf", bbox_inches="tight", pad_inches=0.0, dpi=300)
		close()
	
		push!(lya, coef(fit)[2])
		
end

# lyapunov plotting
using PyPlot

plt.style.use("seaborn-paper")

fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)

axs[1].plot(x, y, ls = "None", ms = 1.0, color = "black", marker = ".", alpha = 0.01)
axs[1].ylim(0,300)
axs[1].ylabel("\$a\$")

axs[2].plot(pvalues, 0.0*pvalues, ls="-", marker = "None", color="red", alpha = 1.0)
axs[2].plot(pvalues, lya, ls = "None", ms = 1.0, color = "black", marker = ".", alpha = 1.0)
axs[2].xlabel("\$ c\$"); 
axs[2].ylabel("\$\\lambda\$")
axs[2].xlim(0,400)
	
tight_layout()
savefig("./combo.pdf", bbox_inches="tight", pad_inches=0.0, dpi=300)
close()
