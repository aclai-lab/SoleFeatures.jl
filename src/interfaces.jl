# -----------------------------------------------------------------------------------------
# abstract types

"""
Abstract supertype for all features selector.

"""
abstract type AbstractFeaturesSelector{T<:AbstractLimiter} end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractFilterBased{T<:AbstractFilterLimiter} <: AbstractFeaturesSelector{T} end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractWrapperBased{T<:AbstractWrapperLimiter} <: AbstractFeaturesSelector{T} end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractEmbeddedBased{T<:AbstractEmbeddedLimiter} <: AbstractFeaturesSelector{T} end

# -----------------------------------------------------------------------------------------
# Abstract filter

abstract type AbstractCorrelationFilter{T<:AbstractFilterLimiter} <: AbstractFilterBased{T} end
abstract type AbstractVarianceFilter{T<:AbstractFilterLimiter} <: AbstractFilterBased{T} end
abstract type AbstractRandomFilter{T<:AbstractFilterLimiter} <: AbstractFilterBased{T} end
abstract type AbstractMeasuresFilter{T<:AbstractFilterLimiter} <: AbstractFilterBased{T} end
abstract type AbstractStatisticalFilter{T<:AbstractFilterLimiter} <: AbstractFilterBased{T} end
abstract type AbstractWindowsFilter{T<:AbstractFilterLimiter} <: AbstractFilterBased{T} end

# -----------------------------------------------------------------------------------------
# AbstractFeaturesSelector

"""
    apply(X, selector)
    apply(X, y, selector)

Return vector containing indicies of suitable attributes from selector.

## ARGUMENTS
- `X::AbstractDataFrame`: samples to evaluate
- `y::AbstractVector{<:Union{String, Symbol}}`: target vector
- `selector::AbstractFeaturesSelector`: applied selector
"""
function apply(
    X::AbstractDataFrame,
    selector::AbstractFeaturesSelector{<:AbstractLimiter}
)::Vector{Integer}
    return error("`apply` for unsupervised selectors not implemented " *
        "for type: $(typeof(selector))")
end

function apply(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::AbstractFeaturesSelector{<:AbstractLimiter}
)::Vector{Integer}
    return error("`apply` for supervised selectors not implemented " *
        "for type: $(typeof(selector))")
end

function limiter(selector::AbstractFeaturesSelector)
    return error("Not implemented for $(typeof(selector))")
end
