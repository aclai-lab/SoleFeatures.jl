# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    FtestFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
    FtestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}

Univariate filter-based feature selection using F-tests.

# Classification (categorical or integer `y`)
Per-feature one-way ANOVA (F-test) to assess mean differences across classes.
Labels that are not integers are level-encoded internally.

# Regression (continuous `y`)
Per-feature simple linear regression; returns F-statistics derived from the
squared correlation with the target.

# Arguments
- `X::AbstractArray{T}`: Matrix of shape (n_samples, n_features)
- `y`:  
  - `AbstractVector` (classification): integer or categorical labels  
  - `AbstractVector{<:AbstractFloat}` (regression): continuous targets

# Fields
- `rank::Vector{Int64}`: Feature indices sorted by descending F-statistic
- `score::Vector{F}`: F-statistic for each feature

# Notes
- Higher F-statistics indicate more discriminative (classification) or more
  explanatory (regression) features.

# Examples
```julia
# Classification
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
filter = FtestFilter(X, y)

# Regression
y_reg = [6.8, 7.7, 3.7, 1.5, 9.2, 9.0]
filter_reg = FtestFilter(X, y_reg)
```
"""
struct FtestFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function FtestFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))
        T isa AbstractFloat             || (X=float.(X))
        rank, score = _f_statistic_classif(X, y)
        new{eltype(score),ClassificationTask,Supervised,Univariate}(rank, score)
    end

    function FtestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
        T isa AbstractFloat             || (X=float.(X))
        rank, score = _f_statistic_regress(X, y)
        new{eltype(score),RegressionTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                            f_statistic classifier                            #
# ---------------------------------------------------------------------------- #
function _f_statistic_classif(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
    nfeatures = size(X,2)
    f_statistic = Vector{T}(undef, nfeatures)

    Threads.@threads for i in axes(X, 2)
        f_statistic[i] = _f_vec(X[:,i], y)
    end

    return sortperm(f_statistic, rev=true), f_statistic
end

function _f_vec(x::AbstractVector{T}, y::AbstractVector) where {T<:Real}
    classes     = unique(y)
    nclasses, n = length(classes), length(x)
    class_sqr   = Vector{T}(undef, nclasses)

    @inbounds for (i, c) in enumerate(classes)
        mask = y .== c
        class_sqr[i] = sum(@view x[mask])^2 / sum(mask)
    end

    grand_sqr  = sum(x)^2 / n

    ss_between = sum(class_sqr[i] for i in 1:nclasses) - grand_sqr
    ss_within  = sum(x.^2) - grand_sqr - ss_between

    ms_between = ss_between / (nclasses - 1)
    ms_within  = ss_within / (n - nclasses)
    
    return iszero(ms_within) ? zero(T) : ms_between / ms_within
end

# ---------------------------------------------------------------------------- #
#                            f_statistic regression                            #
# ---------------------------------------------------------------------------- #
function _f_statistic_regress(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
    _, correlation_coefficient = _r_statistic_regress(X, y)
    
    degrees_of_freedom = length(y) - 2
    corr_coef_squared = correlation_coefficient.^2
    
    f_statistic = corr_coef_squared ./ (1 .- corr_coef_squared) .* degrees_of_freedom

    mask_inf = isinf.(f_statistic)
    f_statistic[mask_inf] .= floatmax(eltype(f_statistic))
    
    mask_nan = isnan.(f_statistic)
    f_statistic[mask_nan] .= 0.0
    
    return sortperm(vec(f_statistic), rev=true), vec(f_statistic)
end

