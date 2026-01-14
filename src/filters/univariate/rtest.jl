# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    RtestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}

A univariate filter-based feature selection method using Pearson's R correlation coefficient.

This filter computes the correlation between each feature and the target variable for regression
tasks. The correlation coefficient measures the linear relationship between each feature and the
target, with values ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).

# Arguments
- `X::AbstractArray{T}`: Matrix of shape (n_samples, n_features)
- `y::AbstractVector{<:AbstractFloat}`: Continuous target vector of length `n_samples`

# Fields
- `rank::Vector{Int64}`: Feature indices sorted by descending absolute correlation
- `score::Vector{<:Real}`: Pearson's R correlation coefficient for each feature

# Notes
- Correlation values lie in the range [-1, 1]
- NaN values (from constant features) are replaced with 0.0
- This is equivalent to univariate linear regression without p-values

# Example
```julia
X = randn(100, 6)
y_reg = rand(100)
filter = RtestFilter(X, y_reg)
```
"""
struct RtestFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function RtestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
        rank, score = _r_statistic_regress(X, y)
        new{eltype(score),RegressionTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                            r_statistic regression                            #
# ---------------------------------------------------------------------------- #
function _r_statistic_regress(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
    n_samples = size(X, 1)
    
    y_centered = y .- mean(y)
    X_means = mean(X, dims=1)
    X_squared_sum = sum(X.^2, dims=1)
    X_norms = sqrt.(X_squared_sum .- n_samples .* X_means.^2)
    
    r_result = vec((y_centered' * X) ./ X_norms ./ LinearAlgebra.norm(y_centered))

    nan_mask = isnan.(r_result)
    r_result[nan_mask] .= 0.0
    
    return sortperm(r_result, rev=true), r_result
end

