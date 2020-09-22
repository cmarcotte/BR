using LinearAlgebra, PyPlot

push!(LOAD_PATH,pwd())
using map
F = map.F
dF= map.dF

function newton!(x,p; K=24, tol=1e-13)

    k = 0
	while norm(F(x,p)) > tol && k < K
		l = sqrt(2k/K)
		x = x .- l * (dF(x,p)\F(x,p))
		k = k + 1
	end
	return x
end

function pltorbit(x,p)
	
	n = length(x)
	
	if n==1
		col="black"
	elseif n==2
		col="red"
	elseif n==3
		col="green"
	elseif n==4
		col="blue"
	else
		col="purple"
	end
	
	J = dF(x,p)+I
	r = F(x,p)
	l = abs(det(J))
	
	if l > 1.0
		mss=1.0
		mrk="."
	else
		mss=3.0
		mrk="."
	end
	
	ax1 = subplot(2,1,1)
	plot(p[1]*ones(size(x)), x, ls="none", ms=mss, color=col, marker=mrk, alpha=1.0)
	xlim(0,400); ylim(0,300)
	ylabel("\$a\$")
	
	ax2 = subplot(2,1,2, sharex = ax1)
	plot([0.0,400.0],[0.0, 0.0], "-r")
	plot(p[1], log(l)/n, ls="none", ms=mss, color=col, marker=mrk)
	xlabel("\$c\$")
	ylabel("\$\\bar\\lambda\$")
	
	tight_layout()
	savefig("./POdiag.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
	
end

# figure
figure(figsize=(5,3))

# PO tolerance
tol = 1e-13

# length of orbit search
for n = 1:4

	# sample space randomly to find more branches
	for c0 in 1.0:1.0:400
	
		# base parameters
		p = [ c0, 	43.54 ]
	
		# sample initial APDs
		for a in 1.0:30.0:300.0
	
			# generate test orbit
			x = zeros(Float64,n)
			x[1] = a
			for m=2:n
				x[m] = f(x[m-1],p)
			end
			
			# solve for periodic orbit
			x = newton!(x,p; K=24)
			
			if n > 1
				for m=2:n
					if isapprox(x[m],x[1])
						x = x[1:(m-1)]
						break
					end
				end
			end
			
			if norm(F(x,p)) < tol && length(x)==n
				pltorbit(x,p)
				println("Orbit found:")
				println("\t p=$(p), x=$(x), l=$(det(dF(x,p)+I))")
			end
				
		end
	
	end

end

close()