include("../DarcyAL_Flat.jl")

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
ranks = distribute_with_mpi(LinearIndices((np,)))

models = generate_distributed_refined_cartesian_models(ranks,4)
main(models;_i_am_main=i_am_main(ranks))
