# ---------------------------------------------------------------------------- #
#                             univariate filters                               #
# ---------------------------------------------------------------------------- #
abstract type AbstractUnivariateFilterBased{T<:AbstractLimiter} <: AbstractFilterBased end

abstract type AbstractVarianceFilter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractRandomFilter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractStatisticalFilter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractChi2Filter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractPearsonCorFilter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractMrMrFilter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractMutualInformationClassif{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractSupLaplacianScore{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractFisherScore{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end
abstract type AbstractIdentityFilter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end

is_univariate(::AbstractUnivariateFilterBased) = true

# ---------------------------------------------------------------------------- #
#                            multivariate filters                              #
# ---------------------------------------------------------------------------- #
abstract type AbstractMultivariateFilterBased <: AbstractFilterBased end
abstract type AbstractCorrelationFilter <: AbstractMultivariateFilterBased end

is_multivariate(::AbstractMultivariateFilterBased) = true

# ---------------------------------------------------------------------------- #
#                            functions definitions                             #
# ---------------------------------------------------------------------------- #
function score(
    X::AbstractDataFrame,
    selector::AbstractUnivariateFilterBased{<:AbstractLimiter}
)
    return error("`score` for unsupervised selectors not implemented " *
        "for type: $(typeof(selector))")
end

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:SoleData.SoleBase.CLabel},
    selector::AbstractUnivariateFilterBased{<:AbstractLimiter}
)
    return error("`score` for supervised selectors not implemented " *
        "for type: $(typeof(selector))")
end

function limiter(selector::AbstractUnivariateFilterBased)
    !hasproperty(selector, :limiter) &&
        throw(ErrorException("`selector` struct not contain `limiter` field"))
    return selector.limiter
end

function apply(
    X::AbstractDataFrame,
    selector::AbstractUnivariateFilterBased
)
    return limit(score(X, selector), limiter(selector))
end

function apply(
    X::AbstractDataFrame,
    y::AbstractVector{<:SoleData.SoleBase.CLabel},
    selector::AbstractUnivariateFilterBased
)
    return limit(score(X, y, selector), limiter(selector))
end