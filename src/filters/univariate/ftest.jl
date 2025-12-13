# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    FtestFilter{F,T,L,D} <: AbstractFilterBased{F,T,L,D}

# Classification
A univariate filter-based feature selection method using One Way ANOVA test.
ANOVA means Analysis of Variance, the main purpose of ANOVA is to test if two or more groups differ
from each other significantly in one or more characteristics. F-test is another name for ANOVA that
only compares the statistical means in groups.

# Regression
Univariate linear regression tests returning F-statistic and p-values.
Quick linear model for testing the effect of a single regressor, sequentially for many regressors.

# Fields
- `rank::Vector{Int64}`: Indices of features sorted by their F-statistic scores in descending order
- `score::Vector{F}`: F-statistic scores for each feature

# Constructors
- `FtestFilter(X::AbstractArray, y::AbstractVector{<:Int})`: For classification tasks
- `FtestFilter(X::AbstractArray, y::AbstractVector{<:AbstractFloat})`: For regression tasks

# Type Parameters
- `F<:Real`: Type of the feature scores
- `T<:AbstractTask`: Task type (ClassificationTask or RegressionTask)
- `L<:AbstractLearning`: Learning paradigm (Supervised)
- `D<:AbstractDimensionality`: Dimensionality type (Univariate)

# Examples
```julia
# Classification
X = rand(100, 10)  # 100 samples, 10 features
y = rand(1:3, 100) # 3 classes
filter = FtestFilter(X, y)

# Regression
y_reg = rand(100)
filter_reg = FtestFilter(X, y_reg)
```

# Notes
- For classification, the F-statistic is computed using one-way ANOVA
- Features with higher F-statistics are more discriminative
- For regression, y must be a float vector values
"""
struct FtestFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function FtestFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))
        rank, score = _f_statistic_classif(X, y)
        new{eltype(score),ClassificationTask,Supervised,Univariate}(rank, score)
    end

    function FtestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
        rank, score = _f_statistic_regress(X, y)
        new{eltype(score),RegressionTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                            f_statistic classifier                            #
# ---------------------------------------------------------------------------- #
function _f_statistic_classif(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
    nclasses = size(X,2)
    f_statistic = Vector{T}(undef, nclasses)

    if nclasses > 10
        Threads.@threads for i in axes(X, 2)
            f_statistic[i] = _f_statistic_classif(X[:,i], y)
        end
    else
        for i in axes(X, 2)
            f_statistic[i] = _f_statistic_classif(X[:,i], y)
        end
    end

    return sortperm(f_statistic, rev=true), f_statistic
end

function _f_statistic_classif(x::AbstractVector{T}, y::AbstractVector) where {T<:Real}
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

