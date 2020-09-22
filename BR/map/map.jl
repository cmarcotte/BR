module map

export f

function f(x,p)
	
	c = p[1]
	d_min = p[2]
	
	d = c - x
	while real(d) < d_min
		d = d + c
	end

	# a = a(d)
	return 258.0 + 125.0*exp(-0.068*(d-d_min)) - 350.0*exp(-0.028*(d-d_min))
	
end


function F(x,p)
	
	r = zeros(eltype(x),length(x))
    # "boundary condition" for PO
    r[1] = f(x[end],p)-x[1]
    # interior nodes
    for n=2:length(x)
        r[n] = f(x[n-1],p)-x[n]
    end
    return r
    
end

function dF(x,p; h=1e-15im)

	J = zeros(eltype(x),length(x),length(x))
    for m = 1:length(x)
        dx = zeros(Float64, length(x))
        dx[m] = 1.0
        J[:,m] = real.((F(x .+ h.*dx,p) .- F(x,p))/h)
    end
    return J
    
end

#		a
u0 = 200.0

#		c		d_min	a_min	d_0		a_0
p0 = [320.0, 	43.54 ]#,	33.0,  300.0,	300.0]

end