push!(LOAD_PATH,pwd())
using map

using PyPlot

plt.style.use("seaborn-paper")

using LinearAlgebra: norm
using LsqFit



function pltt(Cs,Xs,Ls)
	
	# figure
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	
	
	axs[1].plot(Cs, Xs, ls="none", ms=3.0, color="black", marker=".", alpha=0.1)
	axs[1].set_xlim(0,400); 
	axs[1].set_ylim(0,300)
	axs[1].set_ylabel("\$a\$")
	
	axs[2].plot([0.0,400.0],[0.0, 0.0], "-r")
	axs[2].plot(Cs, Ls, ls="none", ms=3.0, color="black", marker=".", alpha=0.1)
	axs[2].set_xlabel("\$c\$")
	axs[2].set_ylabel("\$\\lambda_1\$")
	axs[2].set_ylim([-7,+2])
	axs[2].set_yticks([-6,-4,-2,0,])
	
	tight_layout()
	savefig("./lydiag.pdf", bbox_inches="tight", pad_inches=0.0, dpi=300)
	close()

end

Cs = []
Xs = []
Ls = []

# c range
crange = 1.0:1.0:400.0

# sampled trajectories
X = zeros(Float64,100)
Y = zeros(Float64,100)
d = X .- Y

for c in crange
	
	p = [c, map.p0[2]]
	
	for u0 in 300.0*rand(Float64,30)
		
		X[1] = u0
		Y[1] = u0 + 1e-9
		d[1] = norm(Y[1]-X[1])
		for n in 2:length(X)
			X[n] = f(X[n-1],p)
			Y[n] = f(Y[n-1],p)
			d[n] = norm(X[n] - Y[n])
		end
		
		# subset consisting of exponential growth/decay
		i=1
		q = i
		while d[i] > 1e-15 && d[i] < 1e-1 && i < length(d)
			q = i
			i = i + 1
		end
		
		@. model(x,y) = y[1] + y[2]*x
		fit = curve_fit(model, (1:q), log.(d[1:q]), [d[1],0.0])
		
		push!(Cs,c)
		push!(Xs,X[end])
		push!(Ls,coef(fit)[2])
	
		
	
	end
	
	pltt(Cs,Xs,Ls)
		
end

pltt(Cs,Xs,Ls)
