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


u_exact(x) = VectorValue(0.0, 0.0 )
p_exact(x) = H0 + 0.1*sin(2*π*x[1])*cos(2*π*x[2])

hamiltonian((u,p),dΩ) = sum(∫( 0.5*H0*( u⋅u ) + 0.5*gravity*p*p)dΩ)


function get_initial_condition(model,p_fe::Int,u_exact::Function,p_exact::Function)

  V = TestFESpace(model,ReferenceFE(raviart_thomas,Float64,p_fe);conformity=:Hdiv);
  U = TrialFESpace(V)
  Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,p_fe);conformity=:L2)
  P = TrialFESpace(Q)

  X = MultiFieldFESpace([U,P])

  xh0 = interpolate([u_exact,p_exact],X)
  xh0
end

function change_model_transient_wave(model,p_fe::Int,dir::String,_xh0,
  t0=0.0,tF=2*π,CFL=0.1,
  ls=LUSolver(),return_vtk=false;_i_am_main=true)

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

  xh0 = interpolate(_xh0,X)

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
  nls = GridapSolvers.NonlinearSolvers.NewtonSolver(ls;maxiter=5,atol=1e-14,rtol=1.e-8,verbose=_i_am_main)
  solver = ThetaMethod(nls,dt,0.5)


  uh0,ph0 = xh0


  _i_am_main && mkpath(dir*"/transient_sol")
  writevtk(Ω,dir*"/transient_sol/solT_$(t0).vtu", cellfields=["vel"=>uh0,"p"=>ph0],
    append=false)

  xhF,energy,ts = _transient_wave_equation(
        solver,opT,xh0,dΩ_error,
        dir,Ω,t0,tF,return_vtk;_i_am_main=_i_am_main)


  uhF,phF = xhF
  writevtk(Ω,dir*"/transient_sol/solT_$tF.vtu", cellfields=["vel"=>uhF,"p"=>phF],
          append=false)


  xhF,energy,ts
end


function transient_wave_equation(model,p_fe::Int,_dir::String,u_exact::Function,p_exact::Function,
  simName::String,TF=2*π,CFL=0.1,
  ls=LUSolver(),return_vtk=false;_i_am_main=true)

  dir = _dir*"/$(simName)"
  (_i_am_main && !isdir(dir)) && mkpath(dir)

  xh0 = get_initial_condition(model,p_fe,u_exact,p_exact)
  Es = Float64[]
  Ts = Float64[]

  nadapt = 2
  _dtF = TF/nadapt
  t0 = 0

  for i in collect(1:nadapt)

    ref_coarse_flags = initial_unbalance(model)
    fmodel, fglue = Gridap.Adaptivity.adapt(model,ref_coarse_flags)
    model = fmodel

    tF = _dtF*(i)

    println("adapting model; t0 = $(t0), tF = $(tF)")

    xhF,energy,ts = change_model_transient_wave(model,p_fe,dir,xh0,
      t0,tF,CFL,
      ls,return_vtk;_i_am_main=_i_am_main)

    xh0 = xhF
    t0 = ts[end]
    Es = vcat(Es,energy)
    Ts = vcat(Ts,ts)

  end

  _make_pvd_distributed(dir*"/transient_sol","solT",1)

  relative_energy = map(x->(x-Es[1])/Es[1],Es)

  dir_output = _dir*"/output"
  (_i_am_main && !isdir(dir_output)) && mkpath(dir_output)

  _nnn = num_cells(model)
  output = @strdict _nnn Es relative_energy Ts simName
  _i_am_main && safesave(datadir(dir_output, ("output_$(simName).jld2")), output)

end


function _transient_wave_equation(
  solver,opT,xh0,dΩ_error,
  dir,Ω,
  t0=0.0,tF=2*π,
 return_vtk=false;_i_am_main=true)

  ### NEW FUNCTION HERE
  solT = solve(solver, opT, t0, tF, xh0)

  ## iterate solution
  it = iterate(solT)
  xhF = xh0

  energy = Float64[]
  ts = Float64[]

  E0 = hamiltonian(xh0,dΩ_error)
  push!(energy,E0)
  push!(ts,t0)

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

  EF = hamiltonian(xhF,dΩ_error)
  push!(energy,EF)
  push!(ts,tF)


  xhF, energy, ts
end

n = 8
p_fe = 1
H0, gravity = 1.0, 1.0
ls = LUSolver()
TF = 2*π
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

model = models[2]
name = simName[2]
# for (i,(model,name)) in enumerate(zip(models,simName))
  transient_wave_equation(model,p_fe,dir,u_exact,p_exact,
    name,TF,CFL,LUSolver(),true;_i_am_main=false)
# end


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
ts = df[!,:Ts]


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
