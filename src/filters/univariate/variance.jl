# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    VarianceFilter(X::AbstractArray{T}) where {T<:Real}

Computes the variance of each feature (column) and ranks them in descending order.

# Arguments
- `X::AbstractArray{T}`: Data matrix where rows are samples and columns are features

# Fields
- `rank::Vector{Int64}`: Feature indices sorted by descending absolute correlation
- `score::Vector{<:Real}`: Pearson's R correlation coefficient for each feature

# Example
```julia
X = rand(100, 10)
filter = VarianceFilter(X)
```
"""
struct VarianceFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function VarianceFilter(X::AbstractArray{T}) where {T<:Real}
        rank, score = _variance(X)
        new{eltype(score),AbstractTask,Unsupervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                              variance filter                                 #
# ---------------------------------------------------------------------------- #
function _variance(X::AbstractArray{T}) where {T<:Real}
    v_result = Statistics.var.(eachcol(X))
    return sortperm(v_result, rev=true), v_result
end
