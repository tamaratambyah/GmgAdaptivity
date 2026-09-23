using Gridap
using GridapP4est
using GridapDistributed
using PartitionedArrays
using MPI

u_exact(x) = VectorValue(0.0, x[1] )
p_exact(x) = 1.0 + 0.1*sin(2*π*x[1])

p_fe = 2
n = 16

coarse_model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
ranks = distribute_with_mpi(LinearIndices((np,)))

dmodel = OctreeDistributedDiscreteModel(ranks, coarse_model)
ref_coarse_flags=map(partition(get_cell_gids(dmodel.dmodel)), local_views(dmodel) ) do indices,lmodel
  flags=zeros(Cint,length(indices))
  flags.=refine_flag
end
fmodel, = Gridap.Adaptivity.adapt(dmodel,ref_coarse_flags);


model = dmodel
Ω = Triangulation(model)
qdegree = 2*(p_fe+2)
dΩ = Measure(Ω,qdegree)

# get cellfield of functions
h_cf = CellField(p_exact,Ω)
u_cf = CellField(u_exact, Ω)

# finite element spaces
V = TestFESpace(model,ReferenceFE(raviart_thomas,Float64,p_fe);conformity=:Hdiv);
U = TrialFESpace(V)
Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,p_fe);conformity=:L2)
P = TrialFESpace(Q)

# interpolate functions -- WORKS!
uh0 = interpolate(u_exact,U)
ph0 = interpolate(p_exact,P)

# interpolate cellfields -- FAILS!
interpolate(u_cf,U)
interpolate(h_cf,P)

# Multifield:
X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

interpolate([u_exact,p_exact],X) #--FAILS due to triangulation issue
interpolate([u_cf,h_cf],X) # -- FAILS
