"""
Solves one time step of transient wave equation with crank nicolson, on flat domain.
Test that energy is conserved to machine precision.
"""

using Gridap
using Gridap.Algebra
using Test

u_exact(x) = VectorValue(sin(2*π*x[1])*cos(2*π*x[2]), sin(2*π*x[2])*cos(2*π*x[1]) )
p_exact(x) = H0 + 0.1*sin(2*π*x[1])*cos(2*π*x[2])

energy_exact(x) = 0.5*H0*( u_exact(x)⋅u_exact(x) ) + 0.5*gravity*p_exact(x)
hamiltonian((u,p),dΩ) = sum(∫( 0.5*H0*( u⋅u ) + 0.5*gravity*p*p)dΩ)

n = 8
p_fe = 1
H0, gravity = 1.0, 1.0
ls = LUSolver()
tF = 1.0
CFL = 0.1
dir = @__DIR__

_dt = (1/n)*CFL/p_fe
nsteps = tF/ _dt
dt = tF/floor(nsteps)

model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))

Ω = Triangulation(model)
qdegree = 2*(p_fe+2)
dΩ = Measure(Ω,qdegree)
dΩ_error = Measure(Ω,2*qdegree)

V = TestFESpace(model,ReferenceFE(raviart_thomas,Float64,p_fe);conformity=:Hdiv);
U = TrialFESpace(V)
Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,p_fe);conformity=:L2)
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])
xh0 = interpolate([u_exact,p_exact],X)
uh0,ph0 = xh0
E0 = hamiltonian(xh0,dΩ_error)

biform_u((u,p),(v,q)) = ∫(u⋅v)dΩ - ∫( (dt*0.5*gravity)*divergence(v)*p)dΩ
biform_p((u,p),(v,q)) = ∫( (p*q) )dΩ + ∫( (dt*0.5*H0)*divergence(u)*q)dΩ
liform_u((v,q),(u0,p0)) = ∫(u0⋅v)dΩ + ∫( (dt*0.5*gravity)*divergence(v)*p0)dΩ
liform_p((v,q),(u0,p0)) = ∫( (p0*q) )dΩ - ∫( (dt*0.5*H0)*divergence(u0)*q)dΩ

biform((u,p),(v,q)) =  biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))
liform((v,q)) = liform_u((v,q),xh0) + liform_p((v,q),xh0)

op = AffineFEOperator(biform,liform,X,Y)
A = get_matrix(op)
b = get_vector(op)
ns = numerical_setup(symbolic_setup(ls,A),A)
x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)

xh = FEFunction(X,x)
uh,ph = xh

Eh = hamiltonian(xh,dΩ_error)

relative_conservation_error = (Eh - E0)/E0

@test relative_conservation_error < 1e-10

writevtk(Ω,dir*"/wave_equation_CN.vtu",
  cellfields=["ut0"=>uh0, "pt0"=>ph0,
              "ut1"=>uh, "pt1"=>ph],
  append=false)
