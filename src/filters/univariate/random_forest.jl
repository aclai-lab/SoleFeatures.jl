# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
struct RandomForestInfo <: AbstractFilterInfo
    n_subfeatures       :: Int64
    n_trees             :: Int64
    partial_sampling    :: Float64
    max_depth           :: Int64
    min_samples_leaf    :: Int64
    min_samples_split   :: Int64
    min_purity_increase :: Float64
    rng                 :: AbstractRNG
end

"""
    RandomForestFilter(X::AbstractArray{T}, y::AbstractVector; kwargs...) where {T<:Real}
    RandomForestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}; kwargs...) where {T<:Real}

Univariate feature selection using random forest impurity importance.

Trains a random forest on the input data and ranks features by their 
impurity-based importance scores. Higher importance indicates that the feature
is more discriminative (classification) or more predictive (regression).

# Arguments
- `X::AbstractArray{T}`: Matrix of shape (n_samples, n_features)
- `y::AbstractVector`: 
  - `AbstractVector` (classification): integer or categorical labels
  - `AbstractVector{<:AbstractFloat}` (regression): continuous targets
- `n_subfeatures::Int64=-1`: Number of features to sample at each split 
  (-1 = √n_features for classification, n_features/3 for regression)
- `n_trees::Int64=10`: Number of trees in the forest
- `partial_sampling::Float64=0.7`: Fraction of samples to use per tree
- `max_depth::Int64=-1`: Maximum tree depth (-1 = unlimited)
- `min_samples_leaf::Int64`: Minimum samples required at leaf node
  (default: 1 for classification, 5 for regression)
- `min_samples_split::Int64=2`: Minimum samples required to split a node
- `min_purity_increase::Float64=0.0`: Minimum impurity decrease to split
- `rng::AbstractRNG=Random.TaskLocalRNG()`: Random number generator

# Fields
- `rank::Vector{Int64}`: Feature indices sorted by descending importance
- `score::Vector{<:Real}`: Impurity importance per feature

# Notes
- Uses `DecisionTree.jl` for forest building and importance calculation

# Examples
```julia
# Classification
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
filter = RandomForestFilter(X, y; n_trees=20)

# Regression
X = randn(100, 6)
y_reg = rand(100)
filter_reg = RandomForestFilter(X, y_reg; n_trees=15)
```
"""
struct RandomForestFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}
    info  :: RandomForestInfo

    function RandomForestFilter(X::AbstractArray{T}, y::AbstractVector; kwargs...) where {T<:Real}
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))
        rank, score, info = _random_forest_classifier(X, y; kwargs...)
        new{eltype(score),ClassificationTask,Supervised,Univariate}(rank, score, info)
    end

    function RandomForestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}; kwargs...) where {T<:Real}
        rank, score, info = _random_forest_regression(X, y; kwargs...)
        new{eltype(score),RegressionTask,Supervised,Univariate}(rank, score, info)
    end
end

# ---------------------------------------------------------------------------- #
#                           random_forest classifier                           #
# ---------------------------------------------------------------------------- #
function _random_forest_classifier(
    X                   :: AbstractArray{T},
    y                   :: AbstractVector;
    n_subfeatures       :: Int64       = -1,
    n_trees             :: Int64       = 10,
    partial_sampling    :: Float64     = 0.7,
    max_depth           :: Int64       = -1,
    min_samples_leaf    :: Int64       = 1,
    min_samples_split   :: Int64       = 2,
    min_purity_increase :: Float64     = 0.0,
    rng                 :: AbstractRNG = Random.TaskLocalRNG()
) where {T<:Real}
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
    rf_result = DecisionTree.impurity_importance(forest)
    info      = RandomForestInfo(
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase,
        rng       
    )
    return sortperm(rf_result, rev=true), rf_result, info
end

# ---------------------------------------------------------------------------- #
#                           random_forest regression                           #
# ---------------------------------------------------------------------------- #
function _random_forest_regression(
    X                   :: AbstractArray{T},
    y                   :: AbstractVector{<:AbstractFloat};
    n_subfeatures       :: Int64       = -1,
    n_trees             :: Int64       = 10,
    partial_sampling    :: Float64     = 0.7,
    max_depth           :: Int64       = -1,
    min_samples_leaf    :: Int64       = 5,
    min_samples_split   :: Int64       = 2,
    min_purity_increase :: Float64     = 0.0,
    rng                 :: AbstractRNG = Random.TaskLocalRNG()
) where {T<:Real}
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
    rf_result = DecisionTree.impurity_importance(forest)
    info      = RandomForestInfo(
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase,
        rng       
    )
    return sortperm(rf_result, rev=true), rf_result, info
end

