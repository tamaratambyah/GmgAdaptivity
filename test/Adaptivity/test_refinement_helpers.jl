using Gridap
using GridapP4est
using GridapDistributed
using PartitionedArrays
using MPI
using GmgAdaptivity
using Test

n = 8
coarse_model = CartesianDiscreteModel((0,1,0,1),(n,n),isperiodic=(true,true))

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
ranks = distribute_with_mpi(LinearIndices((np,)))
dmodel = OctreeDistributedDiscreteModel(ranks, coarse_model)

# 1 level of refinement
ref_coarse_flags = initial_unbalance(dmodel)
fmodel, fglue = Gridap.Adaptivity.adapt(dmodel,ref_coarse_flags)

## 2 levels of refinement
ref_coarse_flags = initial_unbalance(fmodel)
afmodel, afglue = Gridap.Adaptivity.adapt(fmodel,ref_coarse_flags)

@test refinement_level(coarse_model) == 0
@test refinement_level(dmodel) == 0
@test refinement_level(fmodel) == 1
@test refinement_level(afmodel) == 2

@test get_n(coarse_model) == 8
@test get_n(dmodel) == 8
@test get_n(fmodel) == 16
@test get_n(afmodel) == 32
