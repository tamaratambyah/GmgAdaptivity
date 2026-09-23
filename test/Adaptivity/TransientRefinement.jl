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

model = dmodel
for counter in collect(1:10)
  if mod(counter,4) == 0
    println("adapt model")
    ref_coarse_flags = initial_unbalance(model)
    fmodel, fglue = Gridap.Adaptivity.adapt(model,ref_coarse_flags)
    model = fmodel
  end
  println("Ref level = ", refinement_level(model))
end
