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
using GridapSolvers

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


# u_exact(x) = VectorValue(sin(2*π*x[1])*cos(2*π*x[2]), sin(2*π*x[2])*cos(2*π*x[1]) )
u_exact(x) = VectorValue(0.0, 0.0 )
p_exact(x) = H0 + 0.1*sin(2*π*x[1])*cos(2*π*x[2])

hamiltonian((u,p),dΩ) = sum(∫( 0.5*H0*( u⋅u ) + 0.5*gravity*p*p)dΩ)


function transient_wave_equation(model,p_fe::Int,_dir::String,u_exact::Function,p_exact::Function,
  simName::String,t0=0.0,tF=2*π,CFL=0.1,
  ls=LUSolver(),return_vtk=false;_i_am_main=true)

  dir = _dir*"/$(simName)"
  (_i_am_main && !isdir(dir)) && mkpath(dir)

  # number of cells per edge after nref levels of uniform refinement of coarse model.
  # used to compute dt below
  _n = get_n(model)

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



  ## transient weak form
  mass(t, (dtu,dtp), (v,q)) = ∫(dtu⋅v)dΩ + ∫( (dtp*q) )dΩ
  res(t,(u,p),(v,q)) =  ∫(H0* divergence(u)*q)dΩ - ∫( gravity* divergence(v)*p)dΩ
  jac(t,(u,p),(du,dp),(v,q)) = res(t,(du,dp),(v,q))
  jac_t(t,(u,p),(dut,dpt),(v,q)) =  ∫( (dut⋅ v) )dΩ + ∫( (dpt*q) )dΩ


  opT = TransientSemilinearFEOperator(mass, res,(jac,jac_t), X, Y, constant_mass=true)

  # transient parameters
  _dt = (1/_n)*CFL/p_fe
  nsteps = tF/ _dt
  dt = tF/floor(nsteps)

  # solve with CN
  nls = GridapSolvers.NonlinearSolvers.NewtonSolver(ls;maxiter=5,atol=1e-14,rtol=1.e-8,verbose=true)
  solver = ThetaMethod(nls,dt,0.5)


  # Initial condition
  xh0 = interpolate([u_exact,p_exact],X)

  ### NEW FUNCTION HERE
  solT = solve(solver, opT, t0, tF, xh0)

  ## iterate solution
  it = iterate(solT)
  xhF = xh0

  uh0,ph0 = xh0
  energy = Float64[]
  ts = Float64[]
  E0 = hamiltonian(xh0,dΩ_error)
  push!(energy,E0)
  push!(ts,t0)

  _i_am_main && mkpath(dir*"/transient_sol")
  writevtk(Ω,dir*"/transient_sol/solT_$(t0).vtu", cellfields=["vel"=>uh0,"p"=>ph0],
    append=false)

  # counter and freq to output solution
  counter, freq = 1, 50
  while !isnothing(it)
    data, state = it
    t, xh = data
    uh,ph = xh
    xhF = xh

    _i_am_main && println("t = $t")

      if mod(counter,freq) == 0 && return_vtk
        Eh = hamiltonian(xh,dΩ_error)
        push!(energy,Eh)
        push!(ts,t)

        writevtk(Ω,dir*"/transient_sol/solT_$t.vtu", cellfields=["vel"=>uh,"p"=>ph],
          append=false)
      end

    counter = counter + 1
    it = iterate(solT, state)
  end

  (typeof(model) <: DiscreteModel) ? make_pvd(dir*"/transient_sol","solT",1) : _make_pvd_distributed(dir*"/transient_sol","solT",1)

  uhF,phF = xhF
  EF = hamiltonian(xhF,dΩ_error)
  push!(energy,EF)
  push!(ts,tF)
  writevtk(Ω,dir*"/transient_sol/solT_$tF.vtu", cellfields=["vel"=>uhF,"p"=>phF],
          append=false)

  relative_energy = map(x->(x-energy[1])/energy[1],energy)

  dir_output = _dir*"/output"
  (_i_am_main && !isdir(dir_output)) && mkpath(dir_output)

  _nnn = num_cells(model)
  output = @strdict _nnn energy relative_energy ts dt simName
  _i_am_main && safesave(datadir(dir_output, ("output_$(simName).jld2")), output)

end


function _transient_wave_equation(model,p_fe::Int,_dir::String,u_exact::Function,p_exact::Function,
  simName::String,t0=0.0,tF=2*π,CFL=0.1,
  ls=LUSolver(),return_vtk=false;_i_am_main=true)


end

n = 8
p_fe = 1
H0, gravity = 1.0, 1.0
ls = LUSolver()
t0, tF = 0.0, 2*π
CFL = 0.1
dir = datadir("TransientWaveEquation_CN")
_i_am_main = true

coarse_model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
ranks = distribute_with_mpi(LinearIndices((np,)))
dmodel = OctreeDistributedDiscreteModel(ranks, coarse_model)
ref_coarse_flags = initial_unbalance(dmodel)
fmodel, = Gridap.Adaptivity.adapt(dmodel,ref_coarse_flags);

## 2 levels of refinement
ref_coarse_flags = initial_unbalance(fmodel)
afmodel, = Gridap.Adaptivity.adapt(fmodel,ref_coarse_flags);

## Gather models into an array
simName = ["coarse_model", "dmodel", "f1model","f2model"]
models = [coarse_model,dmodel,fmodel,afmodel]

for (i,(model,name)) in enumerate(zip(models,simName))
  transient_wave_equation(model,p_fe,dir,u_exact,p_exact,
    name,t0,tF,CFL,LUSolver(),true;_i_am_main=true)
end


# model = fmodel.dmodel.models.item
# name = "fmodel_broken"
#  transient_wave_equation(model,p_fe,dir,u_exact,p_exact,
#     name,t0,tF,CFL,LUSolver(),true;_i_am_main=true)


using DataFrames
using Plots

include(plotsdir("plotools.jl"))

df = collect_results(dir*"/output")
relative_energy = df[!,:relative_energy]
names = df[!,:simName]
ts = df[!,:ts]


plot()
for i in collect(1:length(relative_energy))
  x = ts[i]
  y = relative_energy[i]
  lab = names[i]
  plot!(x,y,lw=3,
    marker=_markers[i],
    ms=6,
    label=lab)
end
plot!(show=true)
plot!(shape=:auto,
    xlabel="time",
    ylabel="(E-E0)/E0",
    xtickfontsize=11,ytickfontsize=11,
    xguidefontsize=12,yguidefontsize=12,
    legendfontsize=10,
    # legend=:bottomright,
    #legend_columns=2,
    framestyle = :box)
savefig(dir*"/energy_comparison.pdf")

############## Log scale
plot()
for i in collect(1:length(relative_energy))
  x = ts[i]
  y = relative_energy[i]
  idx = y .> 0
  lab = names[i]
  plot!(x[idx],y[idx],lw=3,
    marker=markers[i],
    ms = markersize[i],
    label=lab)
end
plot!(show=true)
plot!(shape=:auto,
    yaxis=:log10,
    xlabel="time",
    ylabel="|E-E0|/E0",
    xtickfontsize=11,ytickfontsize=11,
    xguidefontsize=12,yguidefontsize=12,
    legendfontsize=10,
    legend=:bottomright,
    #legend_columns=2,
    framestyle = :box)
savefig(dir*"/energy_comparison_log.pdf")
