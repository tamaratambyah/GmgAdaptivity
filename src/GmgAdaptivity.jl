module GmgAdaptivity

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays
using PartitionedArrays
using GridapDistributed


include("Visualisation.jl")

export make_pvd
export _make_pvd_distributed


end
