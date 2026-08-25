"""
Check energy conservation for the linear wave equation:
  ∂ₜu + ∇p  = 0
  ∂ₜp + ∇⋅u = 0
"""

using Gridap
using Gridap.Algebra
using GridapP4est
using GridapDistributed
using PartitionedArrays
using MPI
using DrWatson
using GmgAdaptivity

function Gridap.CellData.get_triangulation(a::GridapDistributed.DistributedMultiFieldCellField)
  trians = map(get_triangulation,a.field_fe_fun)
  # @check all(map(t -> t === first(trians), trians))
  return first(trians)
end


function initial_unbalance(dmodel::GridapDistributed.DistributedDiscreteModel)

  ref_coarse_flags=map(partition(get_cell_gids(dmodel.dmodel)), local_views(dmodel) ) do indices,lmodel
    flags=zeros(Cint,length(indices))
    flags.=nothing_flag

    cmap = get_cell_map(get_grid(lmodel))
    ref_points = get_cell_ref_coordinates(lmodel)
    coords = lazy_map(evaluate,cmap,ref_points)

    for (i,xy) in enumerate(coords)
      x = map(x->x[1],xy)
      y = map(x->x[2],xy)
      if any(x .> 0.4 ) && any(x .< 0.6 ) && any(y .> 0.4) && any(y .< 0.6)
          flags[i] = refine_flag
      end
      if any(x .< 0.2 )  && any(y .< 0.2)
        flags[i] = refine_flag
      end
      if any(x .> 0.8 )  && any(y .> 0.8)
        flags[i] = refine_flag
      end

    end
    flags
  end
  return ref_coarse_flags
end

u_exact(x) = VectorValue(0.0, 0.0 )
p_exact(x) = H_0 + 0.1*sin(2*π*x[1])*sin(2*π*x[2])

# u_exact(x) = VectorValue(sin(2*π*x[1])*cos(2*π*x[2]), sin(2*π*x[2])*cos(2*π*x[1]) )
# p_exact(x) = H_0 + 0.1*sin(2*π*x[1])*cos(2*π*x[2])


gravity = 1.0
H_0 = 1.0

p_fe = 1
n = 8
_n = n*2 # number of cells per edge after uniform refinement

coarse_model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
ranks = distribute_with_mpi(LinearIndices((np,)))
dmodel = OctreeDistributedDiscreteModel(ranks, coarse_model)
ref_coarse_flags = initial_unbalance(dmodel)
fmodel, = Gridap.Adaptivity.adapt(dmodel,ref_coarse_flags);

_i_am_main = true
ls = LUSolver()
CFL = 0.1
tF = 2*π

_dir = datadir("TransientWaveEquation_u0")
dir = _dir*"/fmodel"
!isdir(dir) && mkpath(dir)
model = fmodel

Ω = Triangulation(model)
qdegree = 2*(p_fe+2)
dΩ = Measure(Ω,qdegree)

V = TestFESpace(model,ReferenceFE(raviart_thomas,Float64,p_fe);conformity=:Hdiv);
U = TrialFESpace(V)
Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,p_fe);conformity=:L2)
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

# Initial condition
xh0 = interpolate([u_exact,p_exact],X)
t0 = 0.0


## transient weak form
mass(t, (dtu,dtp), (v,q)) = ∫(dtu⋅v)dΩ + ∫( (dtp*q) )dΩ
res(t,(u,p),(v,q)) =  ∫(H_0* divergence(u)*q)dΩ - ∫( gravity* divergence(v)*p)dΩ
jac(t,(u,p),(du,dp),(v,q)) = res(t,(du,dp),(v,q))
jac_t(t,(u,p),(dut,dpt),(v,q)) =  ∫( (dut⋅ v) )dΩ + ∫( (dpt*q) )dΩ


opT = TransientSemilinearFEOperator(mass, res,(jac,jac_t), X, Y, constant_mass=true)

# transient parameters
_dt = (1/_n)*CFL/p_fe
nsteps = tF/ _dt
dt = tF/floor(nsteps)

# solve with SSP RK 3
solver = RungeKutta(ls, ls, dt, :EXRK_SSP_3_3)
solT = solve(solver, opT, t0, tF, xh0)

## iterate solution
it = iterate(solT)
xhF = xh0

counter = 1

_i_am_main && mkpath(dir*"/transient_sol")
freq = 50


dΩ_error = Measure(Ω,2*qdegree)


uh0,ph0 = xh0
energy = Float64[]
E0 = sum( ∫(  0.5*H_0*(uh0⋅uh0) + 0.5*gravity*ph0 )dΩ_error)
push!(energy,E0)

writevtk(Ω,dir*"/transient_sol/solT_$(t0).vtu", cellfields=["vel"=>uh0,"p"=>ph0],
  append=false)
while !isnothing(it)
  data, state = it
  t, xh = data
  uh,ph = xh
  xhF = xh

  _i_am_main && println("t = $t")

    if mod(counter,freq) == 0
      Eh = sum( ∫(  0.5*H_0*(uh⋅uh) + 0.5*gravity*ph )dΩ_error)
      push!(energy,Eh)

      writevtk(Ω,dir*"/transient_sol/solT_$t.vtu", cellfields=["vel"=>uh,"p"=>ph],
        append=false)
    end

  counter = counter + 1
  it = iterate(solT, state)
end

# make_pvd(dir*"/transient_sol","solT",1)
_make_pvd_distributed(dir*"/transient_sol","solT",1)

uhF,phF = xhF
EF = sum( ∫(  0.5*H_0*(uhF⋅uhF) + 0.5*gravity*phF )dΩ_error)
push!(energy,EF)

relative_energy = map(x->(x-energy[1])/energy[1],energy)

using DrWatson
simName = "output"
_nnn = num_cells(model)
output = @strdict _nnn energy relative_energy dt
_i_am_main && safesave(datadir(dir, ("$(simName).jld2")), output)



using Plots
plot(1:length(relative_energy),relative_energy)
savefig(dir*"/nonconforming_energy.pdf")


cc = relative_energy

nc = relative_energy



plot()
plot!(1:length(cc),cc,lw=3,marker=:square,ms=5,label="Conforming")
plot!(1:length(nc),nc,lw=2,marker=:circle,label="Non conforming")
plot!(shape=:auto,
    xlabel="time steps",
    ylabel="(E-E0)/E0",
    # xtickfontsize=11,ytickfontsize=11,
    # xguidefontsize=12,yguidefontsize=12,
    # legendfontsize=10,
    # legend=:botleft,
    #legend_columns=2,
    framestyle = :box)
savefig(dir*"/energy_comparison.pdf")
