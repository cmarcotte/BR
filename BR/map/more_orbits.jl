push!(LOAD_PATH,pwd())
using map

using JLD2, FileIO

F = map.F
dF= map.dF

using PyPlot

plt.style.use("seaborn-paper")

using LinearAlgebra

function newton!(x,p; K=24, tol=1e-13)

    k = 0
	while norm(F(x,p)) > tol && k < K
		l = sqrt(2k/K)
		x = x .- l * (dF(x,p)\F(x,p))
		k = k + 1
		if minimum(x) > 0.0 && maximum(x) < 300.0 && !any(isnan.(x))
			#println("\t\tx = $(x)")
		else
			break
		end
	end
	return x
end

function pltt(Cs,Xs,Ls; ftype="png")
	
	# figure
	fig,axs = plt.subplots(2,1,figsize=(4,3), sharex=true)
	
	axs[2].plot([0.0,400.0],[0.0, 0.0], "-r")
	
	for q=3:-1:1
		
		for n in 1:length(Cs)
	
			if length(transpose(Xs[n]))==q
			
				if length(transpose(Xs[n]))==1
					col="black"
				elseif length(transpose(Xs[n]))==2
					col="tab:blue"
				elseif length(transpose(Xs[n]))==3
					col="tab:orange"
				elseif length(transpose(Xs[n]))==4
					col="tab:green"
				elseif length(transpose(Xs[n]))==5
					col="tab:red"
				elseif length(transpose(Xs[n]))==6
					col="tab:purple"
				elseif length(transpose(Xs[n]))==7
					col="tab:brown"
				elseif length(transpose(Xs[n]))==8
					col="tab:pink"
				elseif length(transpose(Xs[n]))==9
					col="tab:olive"
				elseif length(transpose(Xs[n]))==10
					col="tab:cyan"
				else
					col="tab:gray"
				end
				
				if abs(Ls[n]) > 1.0
					mss=3.0
				else
					mss=5.0
				end
					
				axs[1].plot(Cs[n]*ones(size(transpose(Xs[n]))), transpose(Xs[n]), ls="none", ms=mss, color=col, marker=".", alpha=1.0)
				
				axs[2].plot(Cs[n], log(abs(Ls[n]))/length(transpose(Xs[n])), ls="none", ms=mss, color=col, marker=".", alpha=1.0)
			end
		end
	end
	
	axs[1].set_xlim(0,400); 
	axs[1].set_ylim(0,300)
	axs[1].set_ylabel("\$a\$")
	
	axs[2].set_xlabel("\$c\$")
	axs[2].set_ylabel("\$\\bar\\lambda\$")
	axs[2].set_ylim([-7,+2])
	axs[2].set_yticks([-6,-4,-2,0,])
	
	tight_layout()
	savefig("./orbitdiag.$(ftype)", bbox_inches="tight", pad_inches=0.0, dpi=300)
	close()

end



# load existing data; why can't I put this in a try/catch???
Cs = []
Xs = []
Ls = []
try
	global Cs = load("POs.jld2", "Cs")
	global Xs = load("POs.jld2", "Xs")
	global Ls = load("POs.jld2", "Ls")
	println("Loaded `POs.jld2`.")
catch
	println("Could not load `POs.jld2`.")
	error()
end

# prox eps
ep = 3.0

# c range
clim = [40.0,400.0]

# n range
nlim = [4, 9]

# u range
ulim = [30.0, 300.0]

# PO tolerance
tol = 1e-13

while length(Xs) < 10000
	
	# randomly sample space
	c = (clim[1] + (clim[2]-clim[1])*rand(Float64))
	
	n = rand(nlim[1]:nlim[2])
	
	u0= ulim[1] + (ulim[2]-ulim[1])*rand(Float64)
	
	# form p
	p = [c, map.p0[2]]

	# form X
	X = zeros(Float64,n)
	X[1] = u0
	for n in 2:length(X)
		X[n] = f(X[n-1],p)
	end

	# solve for orbit
	X = newton!(X,p; K=24)

	# check prime periodicity
	if n > 1
		for m=2:n
			if isapprox(X[m],X[1])
				X = X[1:(m-1)]
				break
			end
		end
	end

	# clean up solution, check uniqueness
	if norm(F(X,p)) < tol && length(X)==n
	
		if n>1
			X = circshift(X,1-argmin(X))
		end
		isuniq=true
		for m=1:length(X)
			if X[m] < 0.0 || X[m] > 300.0
				isuniq=false
			end
		end
		for m=1:length(Xs)
			if length(X)==length(Xs[m]) && sqrt(sum(([X;c].-[transpose(Xs[m]);Cs[m]]).^2)) < ep
			#isapprox(c,Cs[m]; atol=ep) && isapprox(X, transpose(Xs[m]); atol=n*tol)
				isuniq=false
			end
		end
		if isuniq
			println("Orbit found:")
			println("\tp=$(p), \n\tx=$(X), \n\tl=$(det(dF(X,p)+I))")
			
			push!(Cs,c)
			push!(Xs,transpose(X))
			push!(Ls,det(dF(X,p)+I))

			# print out length of Xs
			println("length(Xs) = $(length(Xs)).\n")

			@save "POs.jld2" Cs Xs Ls
			#pltt(Cs,Xs,Ls; ftype="pdf")
		end

	end
	
end

@save "POs.jld2" Cs Xs Ls
pltt(Cs,Xs,Ls; ftype="pdf")
