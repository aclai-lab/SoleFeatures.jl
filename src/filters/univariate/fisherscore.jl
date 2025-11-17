# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    FisherScoreFilter{T <: AbstractLimiter} <: AbstractFisherScore{T}

A supervised feature selection filter that uses Fisher score to rank features
based on their discriminative power between classes.

The Fisher score measures the ratio of between-class variance to within-class
variance for each feature. Features with higher scores have better class
separation capability.

# Fields
- `limiter::T`: Feature selection limiter controlling how many features to select

# Example
```julia
using SoleFeatures
# Create filter
filter = FisherScoreFilter(TopK(10))
# Apply to data
selected_features = apply_filter(filter, X, y)
```
"""
struct FisherScoreFilter{T <: AbstractLimiter} <: AbstractFisherScore{T}
    limiter::T
    # parameters
end

is_supervised(::AbstractFisherScore)   = true
is_unsupervised(::AbstractFisherScore) = false

FisherScoreRanking(nbest) = FisherScoreFilter(RankingLimiter(nbest, false))
FisherScoreThreshold(; alpha=0.05) = FisherScoreFilter(ThresholdLimiter(alpha, ≤))

# ---------------------------------------------------------------------------- #
#                                fisher score                                  #
# ---------------------------------------------------------------------------- #
# this filter was tested against scikit learn

"""
    fisher_score(X::AbstractMatrix{T}, y::AbstractVector) -> Vector{Int}

Compute Fisher score for feature selection and return feature indices sorted
by their discriminative power (best to worst).

# Arguments
- `X::AbstractMatrix{T}`: Input data matrix of shape `(n_samples, n_features)`
- `y::AbstractVector`: Class labels vector of shape `(n_samples,)`

# Returns
- `Tuple{Vector{Int}, Vector{Float64}}`: A tuple containing:
  - Feature indices sorted in descending order by Fisher score (best first)
  - Corresponding Fisher scores for each feature

# Notes
- Features with very small within-class variance (< 1e-12) are assigned a
  large variance value (1e4) to avoid numerical instability
- Higher Fisher scores indicate features with better class separation

# Examples
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
idxs, score = fisher_score(X, y)
```
"""
function fisher_score(
    X::AbstractMatrix{T},
    y::AbstractVector
)::Tuple{Vector{Int},Vector{Float64}} where {T<:Float64}
    weigths    = _construct_w_fisher(T, y, size(X, 1))
    degrees    = sum(weigths, dims=2)
    tot_degree = sum(degrees)

    tmp = degrees' * X
    t1  = X .* degrees
    t2  = weigths' * X

    within_class_var  = sum(t1 .* X, dims=1) .- (tmp .* tmp) ./ tot_degree
    within_class_var  = ifelse.(within_class_var .< 1e-12, 1e4, within_class_var)
    between_class_var = sum(t2 .* X, dims=1) .- (tmp .* tmp) ./ tot_degree

    lap_score    = 1 .- (between_class_var ./ within_class_var)
    score = vec(1.0 ./ lap_score .- 1.0)
    return sortperm(score), score
end
fisher_score(X::AbstractArray{<:Real}, args...) = fisher_score(Float64.(X), args...)

function _construct_w_fisher(
    ::Type{T},
    y::AbstractVector,
    n::Int64
) where {T<:Float64}
    weights = zeros(T, n, n)
    
    foreach(unique(y)) do label
        idx = findall(==(label), y)
        @inbounds @views weights[idx, idx] .= 1.0 / length(idx)
    end
    
    return weights
end

# ---------------------------------------------------------------------------- #
#                                    score                                     #
# ---------------------------------------------------------------------------- #
function score(
    ::FisherScoreFilter,
    X::AbstractArray{T},
    y::AbstractVector
)::Vector{Float64} where {T<:Real}
    _, score = fisher_score(X, CategoricalArrays.levelcode.(y))
    return score
end

