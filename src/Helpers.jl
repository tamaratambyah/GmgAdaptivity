using Gridap.Adaptivity: AdaptedDiscreteModel, AdaptivityGlue, RefinementGlue,
                         get_adaptivity_glue, get_old_cell_refinement_rules, num_subcells


"""
Returns the refinement level of each cell of model, relative to the conforming
model the adaptation started from, i.e. the number of times the cell's ancestor
was bisected. Cells of a model not obtained via adapt are at level 0.

The levels are recovered recursively from the chain of AdaptedDiscreteModels,
which stores its parent and the AdaptivityGlue.
A new cell K with parent c = n2o[K] gets
    level(K) = level(c) + (num_subcells(refinement_rule(c)) > 1)

For a DistributedDiscreteModel, the levels are computed on each processor.

Only refinement is supported. Glue that includes coarsening throws an error.
"""
function cell_levels(model::DiscreteModel)
  cell_levels!(zeros(Int, num_cells(model)), model)
end

function cell_levels(model::GridapDistributed.DistributedDiscreteModel)
  map(cell_levels, local_views(model))
end

# DiscreteModel is at level 0.
function cell_levels!(levels::AbstractVector{<:Integer}, model::DiscreteModel)
  @assert length(levels) == num_cells(model)
  fill!(levels, 0)
end

# AdaptedDiscreteModel recurvse through parent, then move them one step forward through the glue.
function cell_levels!(levels::AbstractVector{<:Integer}, model::AdaptedDiscreteModel)
  glue = get_adaptivity_glue(model)
  Gridap.Helpers.@notimplementedif !isa(glue, AdaptivityGlue{<:RefinementGlue}) "Only refinement (no coarsening) is supported"
  parent = model.parent
  parent_levels = cell_levels!(similar(levels, num_cells(parent)), parent)

  # new cell to old cell map
  n2o = glue.n2o_faces_map[end]

  # refinement rule of each cell
  rrs = get_old_cell_refinement_rules(glue)

  # for each cell, determine the number of num_subcells in the refinement rule.
  # i.e. if a cell has been refined via bisection, num_subcells(rrs[c]) = 4
  # if a cell has not been refined, num_subcells(rrs[c]) = 1.
  # Hence, we only want to increase the level of the refined cells, where
  # num_subcells(rrs[c]) > 1
  @assert length(levels) == length(n2o)
  for (K, c) in enumerate(n2o)
    levels[K] = parent_levels[c] + (num_subcells(rrs[c]) > 1)
  end
  levels
end


"""
Returns the global maximum refinement level
"""
function refinement_level(model::GridapDistributed.DistributedDiscreteModel)
  clevels = cell_levels(model)
  mm = map(maximum,clevels)

  M = 0
  map(mm) do _m
    if _m >= M
      M = _m
    end
    M
  end

  M
end


function refinement_level(model::DiscreteModel)
  maximum(cell_levels(model))
end

"""
Return coarse model
"""
function get_coarse_model(model::DiscreteModel)
  model
end

function get_coarse_model(model::OctreeDistributedDiscreteModel)
  model.coarse_model
end



"""
Returns the number of cells per edge, after apply nref levels of uniform refinement
to the underlying coarse model.
This function is used to compute the tme step
"""
function get_n(model)
  nref = refinement_level(model)
  n_coarse = num_cells(get_coarse_model(model))

  # number of cells in model after nref amount of bisection
  n_model = n_coarse*(4^nref)

  # number of cells per edge
  _n = sqrt(n_model)
  _n
end
