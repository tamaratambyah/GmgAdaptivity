using Test
TESTCASE = get(ENV, "TESTCASE", "seq")

# MPI tests
if TESTCASE ∈ ("all", "mpi", "mpi-darcy")
   include("DarcyGMG/mpi/runtests.jl")
end
