# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    FisherScoreFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}

Compute Fisher score for feature selection based on class discriminative power.

The Fisher score measures the ratio of between-class variance to within-class
variance for each feature. Features with higher Fisher scores have better
class separation and are considered more discriminative.

# Arguments
- `X::AbstractMatrix{T}`: Input data matrix of shape `(n_samples, n_features)`
- `y::AbstractVector`: Class labels vector of shape `(n_samples,)`

# Fields
- `rank::Vector{Int64}`: Feature indices sorted by Fisher score (descending)
- `score::Vector{Float64}`: Fisher score for each feature

# Example
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
fisherscore = FisherScoreFilter(X, y)
```
"""
struct FisherScoreFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function FisherScoreFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))
        rank, score = _fisher_score(X, y)
        new{eltype(score),ClassificationTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                                fisher score                                  #
# ---------------------------------------------------------------------------- #
function _fisher_score(X::AbstractMatrix{T}, y::AbstractVector) where {T<:Real}
    weigths    = _construct_w_fisher(T, y, size(X, 1))
    degrees    = sum(weigths, dims=2)
    tot_degree = sum(degrees)

    tmp = degrees' * X
    t1  = X .* degrees
    t2  = weigths' * X

    within_class_var  = sum(t1 .* X, dims=1) .- (tmp .* tmp) ./ tot_degree
    within_class_var  = ifelse.(within_class_var .< 1e-12, 1e4, within_class_var)
    between_class_var = sum(t2 .* X, dims=1) .- (tmp .* tmp) ./ tot_degree

    lap_score = 1 .- (between_class_var ./ within_class_var)
    Tout = T <: AbstractFloat ? T : Float32
    score = vec(one(Tout) ./ lap_score .- one(Tout))
    return sortperm(score, rev=true), score
end

function _construct_w_fisher(::Type{T}, y::AbstractVector, n::Int64) where {T<:Real}
    weights = zeros(T <: AbstractFloat ? T : Float32, n, n)
    
    foreach(unique(y)) do label
        idx = findall(==(label), y)
        @inbounds @views weights[idx, idx] .= 1.0 / length(idx)
    end
    
    return weights
end
