push!(LOAD_PATH,pwd())
using map

using JLD2, FileIO

F = map.F
dF= map.dF

using PyPlot
using LinearAlgebra

plt.style.use("seaborn-paper")

spaced = true

if spaced
	figfile="./spacedcyclediag.pdf"
else
	figfile="./cyclediag.pdf"
end

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

# load existing data; why can't I put this in a try/catch???
Cs = load("POs.jld2", "Cs")
Xs = load("POs.jld2", "Xs")
Ls = load("POs.jld2", "Ls")
println("Loaded `POs.jld2`.")

# PO tolerance
tol = 1e-13

# c target
c = 90.0

# figure
figure(figsize=(4,2.5))

# figure cycle length for plotting
totalcyclelength=18

# looping
for n = 1:10
	for m=1:length(Xs)
		if n==length(Xs[m]) && isapprox(c,Cs[m]; atol=2.0)
		
			p = [c, map.p0[2]]
			X = transpose(Xs[m])
			L = Ls[m]
			
			if abs(c-Cs[m]) > 1e-10
				# solve for orbit at exact c using nearby
				X = newton!(X,p; K=24)
				
				# compute stability
				L = det(dF(X,p)+I)
				
			end
			
			# check prime periodicity
			if n > 1
				for mm=2:n
					if isapprox(X[mm],X[1])
						X = X[1:(mm-1)]
						break
					end
				end
			end
			
			#if n==1
			#	lss="none"
			#else
				lss="-"
			#end
	
			if n==1
				col="black"
			elseif n==2
				col="tab:blue"
			elseif n==3
				col="tab:orange"
			elseif n==4
				col="tab:green"
			elseif n==5
				col="tab:red"
			elseif n==6
				col="tab:purple"
			elseif n==7
				col="tab:brown"
			elseif n==8
				col="tab:pink"
			elseif n==9
				col="tab:olive"
			elseif n==10
				col="tab:cyan"
			else
				col="tab:gray"
			end
			
			if abs(L) > 1.0
				mss=6.0
			else
				mss=8.0
			end
			
			if any(X .< 0.0) || any(X .> 300.0)
				break
				
			else
						
				pltX = zeros(Float64,totalcyclelength)
				for m=1:totalcyclelength
					pltX[m] = X[1+mod(m-1,length(X))]
				end
				
				if spaced
					offset = (n-1)*50
				else
					offset = 0
				end
				
				plot(1:totalcyclelength, offset .+ pltX, ls=lss, ms=mss, color=col, alpha=0.3, label="", zorder=11-n)
				plot(1:n, offset .+ X, ls=lss, ms=mss, color=col, marker=".", alpha=1.0, label="$(n)-cycle", zorder=11-n)
				
				tight_layout()
				savefig(figfile, bbox_inches="tight", pad_inches=0.0, dpi=300)
				
				break
			end
		end
	end
	
end

legend(loc="lower left", ncol=3, edgecolor="none", bbox_to_anchor= (0.0, 1.01))
#xlim([0.5,totalcyclelength+0.5])
xticks(collect(1:1:totalcyclelength))
#ylim([0,320])				
if spaced
	yticks([])
end
xlabel("\$n\$")
ylabel("\$a_n\$")
#title("\$c=$(c)\$")

tight_layout()
savefig(figfile, bbox_inches="tight", pad_inches=0.0, dpi=300)
close()
