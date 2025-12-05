# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
# notes for Michy
# useful infos: https://medium.com/data-science/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
# reference: https://github.com/smazzanti/mrmr
"""
    MrMrFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}

Rank features for classification using minimum redundancy maximum relevance (MRMR) algorithm.
The MRMR algorithm finds an optimal set of features that is mutually and maximally dissimilar
and can represent the response variable effectively.
The algorithm minimizes the redundancy of a feature set and maximizes the relevance of a
feature set to the response variable.
The algorithm quantifies the redundancy and relevance using the mutual information of
variables—pairwise mutual information of features and mutual information of a feature and the response.

# Fields
- `limiter::T`: A limiter that defines the selection criterion to be applied to scores.

# Implementation Details
This algorithm is for classification problems.
"""
struct MrMrFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMrMrFilter) = true
is_unsupervised(::AbstractMrMrFilter) = false

# ---------------------------------------------------------------------------- #
#              minimum redundancy maximum relevance classifier                 #
# ---------------------------------------------------------------------------- #
# this filter was tested against python implementation, see notes above
function random_forest() end

function correlation end

function _f_statistic(x::AbstractVector{T}, y::AbstractVector)::T where {T<:Real}
    # One-way ANOVA F-statistic
    classes = unique(y)
    nclasses, n = length(classes), length(x)
    
    stats = Dict(g => Mean() for g in classes)
    @inbounds for (xi, yi) in zip(x, y)
        fit!(stats[yi], xi)
    end
    
    grand_mean = mean(x)
    
    ss_between = sum(nobs(stats[c]) * (value(stats[c]) - grand_mean)^2 for c in classes)
    ss_within = sum((xi - value(stats[yi]))^2 for (xi, yi) in zip(x, y))
    
    ms_between = ss_between / (nclasses - 1)
    ms_within = ss_within / (n - nclasses)
    
    return iszero(ms_within) ? zero(T) : ms_between / ms_within
end

function _f_statistic(X::AbstractArray{T}, y::AbstractVector)::Vector{T}  where {T<:Real}
    nclasses = size(X,2)
    f_result = Vector{T}(undef, nclasses)
    if nclasses > 10
        Threads.@threads for i in axes(X, 2)
            f_result[i] = _f_statistic(X[:,i], y)
        end
    else
        for i in axes(X, 2)
            f_result[i] = _f_statistic(X[:,i], y)
        end
    end
    return f_result
end

function _kolmogorov_smirnov(
    X::AbstractArray{T},
    y::AbstractVector;
    alternative::Symbol=:two-sided',
    method::Symbol=:auto
) where {T<:Real}

end

    # Notes
    # -----
    # There are three options for the null and corresponding alternative
    # hypothesis that can be selected using the `alternative` parameter.

    # - `less`: The null hypothesis is that F(x) >= G(x) for all x; the
    #   alternative is that F(x) < G(x) for at least one x. The statistic
    #   is the magnitude of the minimum (most negative) difference between the
    #   empirical distribution functions of the samples.

    # - `greater`: The null hypothesis is that F(x) <= G(x) for all x; the
    #   alternative is that F(x) > G(x) for at least one x. The statistic
    #   is the maximum (most positive) difference between the empirical
    #   distribution functions of the samples.

    # - `two-sided`: The null hypothesis is that the two distributions are
    #   identical, F(x)=G(x) for all x; the alternative is that they are not
    #   identical. The statistic is the maximum absolute difference between the
    #   empirical distribution functions of the samples.

    # Note that the alternative hypotheses describe the *CDFs* of the
    # underlying distributions, not the observed values of the data. For example,
    # suppose x1 ~ F and x2 ~ G. If F(x) > G(x) for all x, the values in
    # x1 tend to be less than those in x2.

f_statistic()::Function = x, y -> _f_statistic(x, y)
f_statistic(X::AbstractArray, y::AbstractVector) = _f_statistic(X, y)

kolmogorov_smirnov(; kwargs...)::Function = x, y -> _kolmogorov_smirnov(x, y; kwargs...)
kolmogorov_smirnov(X::AbstractArray, y::AbstractVector; kwargs...) = _kolmogorov_smirnov(X, y; kwargs...)
# random_forest(;)::Function      = x -> _random_forest(x; )
# correlation(;)::Function        = x -> _correlation(x; )

function _estimate_mrmr(
    X           :: AbstractArray{T}, 
    y           :: AbstractVector;
    k           :: Int64,
    relevance   :: Base.Callable,
    redundancy  :: Base.Callable,
    denominator :: Base.Callable
) where {T<:Real}
    relevance_result = relevance(X, y)

end

function mrmr_classif(
    X           :: AbstractArray{T}, 
    y           :: AbstractVector;
    k           :: Int64 =size(X, 2),
    relevance   :: Base.Callable=f_statistic, # f_statistic, kolmogorov_smirnov, random_forest
    redundancy  :: Base.Callable=correlation, # correlation
    denominator :: Base.Callable=mean         # mean, max
# )::Vector{T} where {T<:Real}
) where {T<:Real}
    return _estimate_mrmr(X, y; k, relevance, redundancy, denominator)
end
mrmr_classif(X::AbstractArray{<:Real}, args...; kwargs...) = 
    mrmr_classif(Float64.(X), args...; kwargs...)

# ---------------------------------------------------------------------------- #
#                                    score                                     #
# ---------------------------------------------------------------------------- #
function score(
    ::MrMrFilter,
    X     :: AbstractArray{T},
    y     :: AbstractVector;
    w     :: Union{AbstractVector{T}, Nothing}=nothing,
    prior :: Symbol
)::Vector{T} where {T<:Real}
    y isa AbstractVector{<:Int} || (y=CategoricalArrays.levelcode.(y))
    return mrmr(X, y)
end