using Plots
using ODEInterfaceDiffEq: radau, remake

using Bifurcations
using Bifurcations: LimitCycleProblem

using DifferentialEquations

using Setfield

using DynamicalSystems, DifferentialEquations
using ODEInterfaceDiffEq
using Plots

function ab(C,V)
	# eq (13) from original paper
	return (C[1]*exp(C[2]*(V+C[3]))+C[4]*(V+C[5]))/(exp(C[6]*(V+C[3]))+C[7])
end

function BR!(dx,x,p,t)

	# following https://doi.org/10.1016/0167-2789(84)90283-5

	# p = [C, A, f]
	
	# Harmonic oscillator variables, y & z, are last
	#								v  v
	# x = [V, Ca, x, m, h, j, d, f, y, z]
	
	V = -84.57375612226087#x[1]
		
	# these from Beeler & Reuter table:
	#"""
	ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],V)
	bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],V)
	am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],V)
	bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],V)
	ah = ab([ 0.126 ,-0.25 , 77.0, 0.0, 0.0, 0.0  , 0.0],V)
	bh = ab([ 1.7   , 0.0  , 22.5, 0.0, 0.0,-0.082, 1.0],V)
	aj = ab([ 0.055 ,-0.25 , 78.0, 0.0, 0.0,-0.2  , 1.0],V)
	bj = ab([ 0.3   , 0.0  , 32.0, 0.0, 0.0,-0.1  , 1.0],V)
	ad = ab([ 0.095 ,-0.01 , -5.0, 0.0, 0.0,-0.072, 1.0],V)
	bd = ab([ 0.07  ,-0.017, 44.0, 0.0, 0.0, 0.05 , 1.0],V)
	af = ab([ 0.012 ,-0.008, 28.0, 0.0, 0.0, 0.15 , 1.0],V)
	bf = ab([ 0.0065,-0.02 , 30.0, 0.0, 0.0,-0.2  , 1.0],V)
	#"""
	
	V=x[1]
	
	IK=(exp(0.08*(V+53.0)) + exp(0.04*(V+53.0)))
	IK=4.0*(exp(0.04*(V+85.0)) - 1.0)/IK
	IK=IK+0.2*(V+23.0)/(1.0-exp(-0.04*(V+23.0)))
	IK=0.35*IK
	Ix = x[3]*0.8*(exp(0.04*(V+77.0))-1.0)/exp(0.04*(V+35.0))
	INa = (4.0*x[4]*x[4]*x[4]*x[5]*x[6] + 0.003)*(V-50.0)
	Is = 0.09*x[7]*x[8]*(V+82.3+13.0287*log(x[2]))
	
	# this is the forcing current
	I = p[2]*x[9]
	
	# BR dynamics
	dx[1] = (I - IK - Ix - INa - Is)/p[1]
	dx[2] = -10^-7 * Is + 0.07*(10^-7 - x[2])
	dx[3] = ax*(1.0 - x[3]) - bx*x[3]
	dx[4] = am*(1.0 - x[4]) - bm*x[4]
	dx[5] = ah*(1.0 - x[5]) - bh*x[5]
	dx[6] = aj*(1.0 - x[6]) - bj*x[6]
	dx[7] = ad*(1.0 - x[7]) - bd*x[7]
	dx[8] = af*(1.0 - x[8]) - bf*x[8]
	
	# nonlinear oscillator dynamics
	ω = 2*pi*p[3]/1000.0
	dx[9] = ω*x[10] + x[9] *(1.0 - x[9]^2 - x[10]^2)
	dx[10]=-ω*x[9]  + x[10]*(1.0 - x[9]^2 - x[10]^2)
	
	return nothing

end

# time in ms
tspan = [0.0, 5000.0]

# initial condition
u0 = [-84.0, 10^-7, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.0, 1.0]

# parameters
#	 uF/cm2    uA/cm2       Hz
p = [1.0, 		0.0, 		1.0]

# define V90 threshold
V90 = -75.0

# make problem
ode = ODEProblem(BR!,u0,(0.0,5000.0),p)

sol = solve(ode, radau(), reltol=1e-13, abstol=1e-13)
u0 = sol[end]; u0[9] = 0.0; u0[10] = 1.0;

p[2] = 2.3

# Create an ODEProblem and solve it:
ode = remake(
    ode,
    p = p,
    u0 = u0,
    tspan = (0.0, 5000.0),
)
sol = solve(ode, radau(), reltol=1e-13, abstol=1e-13)

plt_ode = plot(sol, vars=(0,1), tspan=(0.0, 5000.0))

# Let's find a point (approximately) on the limit cycle and its period:
using Roots: find_zero
t0 = find_zero((t) -> sol(t)[1] - V90, 3500.0)
t1 = find_zero((t) -> sol(t)[1] - V90, t0 + 1000.0/p[3])
x0 = sol(t0)
@assert all(isapprox.(x0, sol(t1); rtol=1e-2))
x0
t1-t0

# Then a LimitCycleProblem can be constructed from the ode.
num_mesh = 50
degree = 5
t_domain = ((t1-t0)/2,2*(t1-t0))  # so that it works with this `num_mesh` / `degree`
prob = LimitCycleProblem(
    ode, (@lens _[3]), t_domain,
    num_mesh, degree;
    x0 = x0,
    l0 = t1 - t0,
    de_args = [radau()],
)

# As the limit cycle is only approximately specified, solver option start_from_nearest_root = true must be passed to start continuation:
solver = init(
    prob;
    start_from_nearest_root = true,
    max_branches = 2,
    verbose = true
)
@time solve!(solver)

# By default, plot_state_space plots limit cycles colored by its period:
plt_lc = plot_state_space(solver,(:x => 9, :y => 1, :color => :period))
plot!(xlabel="I/A",ylabel="V")
savefig("./continuation2.pdf")
