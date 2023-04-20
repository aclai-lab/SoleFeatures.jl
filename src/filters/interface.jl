# =========================================================================================
# Univariate filters

abstract type UnivariateFilterBased{T<:AbstractLimiter} <: AbstractFilterBased end

abstract type AbstractVarianceFilter{T<:AbstractLimiter} <: UnivariateFilterBased{T} end
abstract type AbstractRandomFilter{T<:AbstractLimiter} <: UnivariateFilterBased{T} end
abstract type AbstractStatisticalFilter{T<:AbstractLimiter} <: UnivariateFilterBased{T} end
abstract type AbstractChi2Filter{T<:AbstractLimiter} <: UnivariateFilterBased{T} end

is_univariate(::UnivariateFilterBased) = true

function score(
    X::AbstractDataFrame,
    selector::UnivariateFilterBased{<:AbstractLimiter}
)
    return error("`score` for unsupervised selectors not implemented " *
        "for type: $(typeof(selector))")
end

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::UnivariateFilterBased{<:AbstractLimiter}
)
    return error("`score` for supervised selectors not implemented " *
        "for type: $(typeof(selector))")
end

function limiter(selector::UnivariateFilterBased)
    !hasproperty(selector, :limiter) &&
        throw(ErrorException("`selector` struct not contain `limiter` field"))
    return selector.limiter
end

function apply(
    X::AbstractDataFrame,
    selector::UnivariateFilterBased
)
    return limit(score(X, selector), limiter(selector))
end

function apply(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::UnivariateFilterBased
)
    return limit(score(X, y, selector), limiter(selector))
end

# =========================================================================================
# Multivariate filters

abstract type MultivariateFilterBased <: AbstractFilterBased end

abstract type AbstractCorrelationFilter <: MultivariateFilterBased end

is_multivariate(::MultivariateFilterBased) = true
