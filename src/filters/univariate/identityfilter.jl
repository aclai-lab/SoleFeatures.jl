# ---------------------------------------------------------------------------- #
#                                 identity filter                              #
# ---------------------------------------------------------------------------- #
"""
    IdentityFilter(X::AbstractArray{T}, [y::AbstractVector]) where {T<:Real}

An identity filter that assigns equal scores (1.0) to all features, effectively
performing no feature selection. This filter serves as a baseline or placeholder
when no actual feature selection is desired.

The filter supports both supervised and unsupervised modes:
- Supervised (classification): Requires `X` and integer class labels `y`
- Supervised (regression): Requires `X` and continuous target values `y`
- Unsupervised: Requires only `X`

# Arguments
- `X::AbstractArray{T}`: Input data matrix of shape `(n_samples, n_features)`
- `y::AbstractVector`: (Optional) Target vector for supervised tasks
  - Integer vector for classification tasks
  - Float vector for regression tasks
  - Omit for unsupervised tasks

# Fields
- `rank::Vector{Int64}`: Feature indices in original order (no reordering)
- `score::Vector{Float64}`: Uniform scores of 1.0 for all features

# Examples
```julia
# Supervised classification
X = [1 2 3; 4 5 6; 7 8 9]
y = [1, 0, 1]
filter = IdentityFilter(X, y)

# Unsupervised
filter_unsup = IdentityFilter(X)
```
"""
struct IdentityFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function IdentityFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))
        rank, score = y, ones(T, length(y))
        new{eltype(score),ClassificationTask,Supervised,Univariate}(rank, score)
    end

    function IdentityFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
        rank, score = y, ones(T, length(y))
        new{eltype(score),RegressionTask,Supervised,Univariate}(rank, score)
    end

    function IdentityFilter(X::AbstractArray{T}) where {T<:Real}
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))
        rank, score = y, ones(T, length(y))
        new{eltype(score),nothing,Unsupervised,Univariate}(rank, score)
    end
end
