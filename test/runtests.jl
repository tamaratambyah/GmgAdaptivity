using Test
TESTCASE = get(ENV, "TESTCASE", "seq")

# Sequential tests
if TESTCASE ∈ ("all", "seq", "seq-darcy")
   include("DarcyGMG/seq/runtests.jl")
end

# MPI tests
if TESTCASE ∈ ("all", "mpi", "mpi-darcy")
   include("DarcyGMG/mpi/runtests.jl")
end
