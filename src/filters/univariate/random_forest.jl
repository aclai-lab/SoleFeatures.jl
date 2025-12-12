# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    RandomForestFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}


"""
struct RandomForestFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMrMrFilter) = true
is_unsupervised(::AbstractMrMrFilter) = false

# ---------------------------------------------------------------------------- #
#                                 random_forest                                #
# ---------------------------------------------------------------------------- #
function _random_forest(
    X::AbstractArray{T},
    y::AbstractVector;
    n_subfeatures::Int64=-1,
    n_trees::Int64=100,
    partial_sampling::Float64=0.7,
    max_depth::Int64=5,
    min_samples_leaf::Int64=5,
    min_samples_split::Int64=2,
    min_purity_increase::Float64=0.0,
    rng::Union{AbstractRNG,Int64}=0
)::Vector{T} where {T<:Real}
    forest = DecisionTree.build_forest(
        y, X,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase;
        rng)
    return DecisionTree.impurity_importance(forest)
end

random_forest()::Function = x, y -> _random_forest(x, y)
random_forest(X::AbstractArray, y::AbstractVector) = _random_forest(X, y)

# ---------------------------------------------------------------------------- #
#                                    score                                     #
# ---------------------------------------------------------------------------- #
function score(
    ::RandomForestFilter,
    X     :: AbstractArray{T},
    y     :: AbstractVector;
    kwargs...
)::Vector{T} where {T<:Real}
    y isa AbstractVector{<:Int} || (y=CategoricalArrays.levelcode.(y))
    return mrmr_classif(X, y; kwargs...)
end