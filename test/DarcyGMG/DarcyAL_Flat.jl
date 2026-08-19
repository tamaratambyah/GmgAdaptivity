
"""
Solve the Darcy problem on periodic meshes
u + ∇p  = f
    ∇⋅u = g
"""

using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra, Gridap.Adaptivity
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver
using MPI
using GridapP4est
using DrWatson
using GridapGeosciences

#### Note, adapt not reliable for periodic serial models. Use distributed models instead
# function adapt_model(model::UnstructuredDiscreteModel)
#   ref_model = refine(model)
#   adaptivity_glue = get_adaptivity_glue(ref_model)
#   return ref_model, adaptivity_glue
# end

function generate_distributed_refined_cartesian_models(ranks,
                                        n_ref_lvls)
  models = Vector{OctreeDistributedDiscreteModel}(undef,n_ref_lvls-1)

  for n_ref in n_ref_lvls:-1:2 # stop at 2^2 = 4 for periodic
    n = 2^n_ref
    coarse_model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))
    dmodel = OctreeDistributedDiscreteModel(ranks, coarse_model)
    models[n_ref-1] = dmodel
  end
  models
end

function GridapGeosciences.ConvergenceTools.nref(x::OctreeDistributedDiscreteModel)
  println("WARNING: Computing nref for CartesianDiscreteModel")
  Int(log2(sqrt(num_cells(x))))
end

function adapt_model(model::OctreeDistributedDiscreteModel)
  cell_partition=get_cell_gids(model.dmodel)
  ref_flags=map(partition(cell_partition)) do indices
      flags=zeros(Cint,length(indices))
      flags.=refine_flag
  end
  ref_model, adaptivity_glue = Gridap.Adaptivity.adapt(model,ref_flags)
  return ref_model, adaptivity_glue
end

function get_patch_smoothers(sh,biform,qdegree)
  nlevs = num_levels(sh)
  smoothers = map(view(sh,1:nlevs-1)) do shl
    model = get_model(shl)
    ptopo = Geometry.PatchTopology(ReferenceFE{0},model)
    space = get_fe_space(shl)
    Ω  = Geometry.PatchTriangulation(model,ptopo)
    dΩ = Measure(Ω,qdegree)
    ap = (u,v) -> biform(u,v,dΩ)
    solver = PatchBasedSmoothers.PatchSolver(
      ptopo, space, space, ap;
      assembly = :star,
      collect_factorizations = true,
      is_nonlinear = false
    )
    return RichardsonSmoother(solver,10,0.2)
  end
  return smoothers
end

function get_block_jacobi_smoothers(sh)
  nlevs = num_levels(sh)
  smoothers = map(view(sh,1:nlevs-1)) do shl
    model = get_model(shl)
    ptopo = Geometry.PatchTopology(ReferenceFE{0},model)
    space = get_fe_space(shl)
    solver = PatchBasedSmoothers.BlockJacobiSolver(space, ptopo; assembly=:star)
    return RichardsonSmoother(solver,10,0.2)
  end
  return smoothers
end

function get_jacobi_smoothers(mh)
  nlevs = num_levels(mh)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),nlevs-1)
  level_parts = view(get_level_parts(mh),1:nlevs-1)
  return HierarchicalArray(smoothers,level_parts)
end


function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end

function Gridap.CellData.get_triangulation(a::GridapDistributed.DistributedMultiFieldCellField)
  trians = map(get_triangulation,a.field_fe_fun)
  # @check all(map(t -> t === first(trians), trians))
  return first(trians)
end

u_exact(x) = VectorValue(sin(2*π*x[1])*cos(2*π*x[2]), sin(2*π*x[2])*cos(2*π*x[1]) )
p_exact(x) = 1 + 0.1*sin(2*π*x[1])*cos(2*π*x[2])



function darcy_gmg_flat(dmodel::OctreeDistributedDiscreteModel,
  p_fe::Int,dir::String,u_exact::Function,p_exact::Function,γ=1.0,return_vtk=false;_i_am_main=true)

  fmodel,glue = adapt_model(dmodel)
  mh = ModelHierarchy([fmodel.dmodel,dmodel.dmodel])


  model = get_model(mh,1)
  Ω = Triangulation(model)
  qdegree = 2*(p_fe+2)
  dΩ = Measure(Ω,qdegree)

  tests_u = TestFESpace(mh,ReferenceFE(raviart_thomas,Float64,p_fe);conformity=:Hdiv);
  trials_u = TrialFESpace(tests_u);
  U = get_fe_space(trials_u,1)
  V = get_fe_space(tests_u,1)
  Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,p_fe);conformity=:L2)

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  f(x) = u_exact(x) + ∇(p_exact)(x)
  g(x) = (∇⋅u_exact)(x)
  biform_u(u,v,dΩ) = ∫(u⋅v)dΩ + ∫(γ*(divergence(u)*divergence(v)) )dΩ
  biform((u,p),(v,q),dΩ) = ( biform_u(u,v,dΩ)
                            - ∫(divergence(v)*p)dΩ
                            + ∫(divergence(u)*q)dΩ
                            )
  liform((v,q),dΩ) = ∫(v⋅f)dΩ + ∫(q*g)dΩ + ∫( γ*divergence(v)*g  )dΩ

  a(u,v) = biform(u,v,dΩ)
  l(v) = liform(v,dΩ)
  op = AffineFEOperator(a,l,X,Y)
  A, b = get_matrix(op), get_vector(op);

  #### solvers
  biforms = map(mhl -> get_bilinear_form(mhl,biform_u,qdegree),mh)
  smoothers = get_patch_smoothers(tests_u,biform_u,qdegree)
  # smoothers =   get_block_jacobi_smoothers(tests_u)
  # smoothers =  get_jacobi_smoothers(mh) #Fill(RichardsonSmoother(JacobiLinearSolver(),10,0.2),num_levels(tests_u)-1)

  prolongations = setup_prolongation_operators(tests_u,qdegree;mode=:residual)
  restrictions = setup_restriction_operators(
    tests_u,qdegree;mode=:residual,solver=CGSolver(JacobiLinearSolver())
  )

  gmg = GMGLinearSolver(
    trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=LUSolver(),
    maxiter=20,mode=:preconditioner,verbose=_i_am_main,
    atol=1.0e-14, rtol=1.0e-08
  )

  ##### solvers for the blocks of the preconditioner
  # solver_u = LUSolver()
  # solver_u = CGSolver(JacobiLinearSolver();maxiter=1000,atol=1e-14,rtol=1.e-8,verbose=1,name="CG_velocity")
  solver_u = gmg

  # solver_p = LUSolver()
  solver_p = CGSolver(JacobiLinearSolver();maxiter=1000,atol=1e-14,rtol=1.e-8,verbose=1,name="CG_pressure")


  #### preconditioner
  bblocks  = [LinearSystemBlock() LinearSystemBlock();
              LinearSystemBlock() BiformBlock((p,q) -> ∫( (-1.0/γ)*p*q)dΩ,Q,Q)]
  coeffs = [1.0 1.0;
            0.0 1.0]

  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  # P = JacobiLinearSolver()

  ##### Preconditioned external solver
  ls = FGMRESSolver(20,P;maxiter=1000,atol=1e-14,rtol=1.e-10,verbose=true)
  # ls = GMRESSolver(40;Pr=JacobiLinearSolver(),Pl=nothing,maxiter=2000,rtol=1.e-8,verbose=true)
  # ls = LUSolver()
  ns = numerical_setup(symbolic_setup(ls,A),A)

  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)

  xh = FEFunction(X,x)
  uh,ph = xh

  dΩ_error = Measure(Ω,2*qdegree)
  eu = uh-u_exact
  eu_l2 = sqrt(sum(∫( eu⋅eu )dΩ_error )) # 1e-8

  ep = ph-p_exact
  ep_l2 = sqrt(sum(∫( ep⋅ep )dΩ_error ))
  ep_l2 ≈ sum(∫(p_exact)dΩ) # O(1)

  lvl = nref(dmodel)

  if return_vtk
    writevtk(Ω,dir*"/Darcy_nref$(lvl)_p$(p_fe)",
          cellfields= ["uh"=>uh, "ph"=>ph, "eu"=>eu, "ep"=>ep, "u"=>u_exact, "p"=>p_exact, "g"=>g],
          append=false)
  end



  # return eu_l2, ep_l2, false

  u_iters = solver_u.log.num_iters
  p_iters = solver_p.log.num_iters
  kylov_iters = ls.log.num_iters
  return eu_l2, ep_l2, u_iters, p_iters, kylov_iters

end


################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[1],_i_am_main=true)
  dir = @__DIR__
  γ = 1
  p_convergence_auto_test(ps,models,darcy_gmg_flat,dir,u_exact,p_exact,γ;_i_am_main=_i_am_main)
end




################################################################################
#### Launcher for prepare jobs
################################################################################
function launch(ranks,n_ref,p_fe::Int,γ,dir= @__DIR__,return_vtk=0)
  n = 2^n_ref
  _i_am_main = i_am_main(ranks)

  dir_convergence = dir*"/convergence"
  (_i_am_main && !isdir(dir_convergence)) && mkpath(dir_convergence)

  coarse_model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))
  dmodel = OctreeDistributedDiscreteModel(ranks, coarse_model)

  # eu_l2, ep_l2, _ = darcy_gmg_flat(dmodel,p_fe,dir,u_exact,p_exact,γ,Bool(return_vtk);_i_am_main=_i_am_main)
  eu_l2, ep_l2, u_iters, p_iters, kylov_iters = darcy_gmg_flat(dmodel,p_fe,dir,u_exact,p_exact,γ,Bool(return_vtk);_i_am_main=_i_am_main)

  lvl = nref(dmodel)
  simName = "Darcy_nref$(lvl)_p$(p_fe)_gamma$(γ)"
  output = @strdict eu_l2 ep_l2 p_fe lvl n γ u_iters  p_iters  kylov_iters
  _i_am_main && safesave(datadir(dir_convergence, ("$(simName).jld2")), output)

end


MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
ranks = distribute_with_mpi(LinearIndices((np,)))


p_fe = 1
n_ref = 4
γ = 1
dir = datadir("DarcyAL_Flat_GMG_patch")

for _n_ref in collect(2:n_ref+1)
  launch(ranks,_n_ref,p_fe,γ,dir)
end
