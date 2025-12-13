# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    RtestFilter{F,T,L,D} <: AbstractFilter{F,T,L,D}

A univariate filter-based feature selection method using Pearson's R correlation coefficient.

This filter computes the correlation between each feature and the target variable for regression
tasks. The correlation coefficient measures the linear relationship between each feature and the
target, with values ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).

# Fields
- `rank::Vector{Int64}`: Indices of features sorted by their absolute correlation scores in descending order
- `score::Vector{F}`: Pearson's R correlation coefficients for each feature

# Constructor
- `RtestFilter(X::AbstractArray, y::AbstractVector{<:AbstractFloat})`: For regression tasks

# Type Parameters
- `F<:Real`: Type of the feature scores
- `T<:AbstractTask`: Task type (RegressionTask)
- `L<:AbstractLearning`: Learning paradigm (Supervised)
- `D<:AbstractDimensionality`: Dimensionality type (Univariate)

# Examples
```julia
# Regression task
X = rand(100, 10)  # 100 samples, 10 features
y = rand(100)      # continuous target
filter = RtestFilter(X, y)
```

# Notes
- Features with higher absolute correlation values are more predictive
- Correlation values lie in the range [-1, 1]
- NaN values (from constant features) are replaced with 0.0
- This is equivalent to univariate linear regression without p-values
- Recommended for identifying linear relationships between features and target
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
    
    f_result = vec((y_centered' * X) ./ X_norms ./ LinearAlgebra.norm(y_centered))

    nan_mask = isnan.(f_result)
    f_result[nan_mask] .= 0.0
    
    return sortperm(f_result, rev=true), f_result
end

