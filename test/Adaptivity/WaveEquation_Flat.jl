"""
Check energy conservation for the linear wave equation:
  u + ∇p  = f
  p + ∇⋅u = g
"""

using Gridap
using Gridap.Algebra
using GridapP4est
using GridapDistributed
using PartitionedArrays
using MPI
using DrWatson

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

u_exact(x) = VectorValue(sin(2*π*x[1])*cos(2*π*x[2]), sin(2*π*x[2])*cos(2*π*x[1]) )
p_exact(x) = H_0 + 0.1*sin(2*π*x[1])*cos(2*π*x[2])

# u_exact(x) = VectorValue(x[2]*(1-x[2]), x[1]*(1-x[1]) )
# p_exact(x) = H_0 + 0.1*x[1]*(1-x[1])


energy_exact(x) = 0.5*H_0*( u_exact(x)⋅u_exact(x) ) + 0.5*gravity*p_exact(x)

gravity = 1.0
H_0 = 1.0

p_fe = 2
n = 16
dir = datadir("Wave_Energy_Conservation")
!isdir(dir) && mkdir(dir)
coarse_model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
ranks = distribute_with_mpi(LinearIndices((np,)))
dmodel = OctreeDistributedDiscreteModel(ranks, coarse_model)
ref_coarse_flags = initial_unbalance(dmodel)
fmodel, = Gridap.Adaptivity.adapt(dmodel,ref_coarse_flags);

  writevtk(Triangulation(dmodel),dir*"/fmodel",
        append=false)

_i_am_main = true
ls = LUSolver()

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

f(x) = u_exact(x) + ∇(p_exact)(x)
g(x) = p_exact(x) + (∇⋅u_exact)(x)
biform_u((u,p),(v,q)) = ∫(u⋅v)dΩ - ∫( gravity* divergence(v)*p)dΩ
biform_p((u,p),(v,q)) = ∫( (p*q) )dΩ + ∫(H_0* divergence(u)*q)dΩ
biform((u,p),(v,q)) =  biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))
liform((v,q)) = ∫(v⋅f)dΩ + ∫(q*g)dΩ

op = AffineFEOperator(biform,liform,X,Y)
A = get_matrix(op)
b = get_vector(op)
ns = numerical_setup(symbolic_setup(ls,A),A)
x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)

xh = FEFunction(X,x)
uh,ph = xh


dΩ_error = Measure(Ω,2*qdegree)
energy_h = 0.5*H_0*(uh⋅uh) + 0.5*gravity*ph

E0 = sum(∫(  energy_exact )dΩ_error)
Eh = sum(∫( energy_h )dΩ_error)

(Eh - E0)/E0

eu = uh-u_exact
eu_l2 = sqrt(sum(∫( eu⋅eu )dΩ_error )) # 1e-8

ep = ph-p_exact
ep_l2 = sqrt(sum(∫( ep⋅ep )dΩ_error ))

ee = energy_h-energy_exact
ee_l2 =  sqrt(sum(∫( ee⋅ee )dΩ_error ))

lvl = 2#nref(dmodel)

# if return_vtk
  writevtk(Ω,dir*"/fmodel",
        cellfields= ["uh"=>uh, "ph"=>ph, "eu"=>eu, "ep"=>ep, "u"=>u_exact,
        "p"=>p_exact, "g"=>∇⋅u_exact,
        "Eh"=>energy_h,"E_ex"=>energy_exact, "energy_e"=>ee],
        append=false)
# end
