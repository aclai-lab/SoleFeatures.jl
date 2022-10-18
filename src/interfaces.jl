# -----------------------------------------------------------------------------------------
# abstract types

"""
Abstract supertype for all features selector.

A concrete subtype of AbstractFeaturesSelector should always provide functions
[`transform`](@ref) and [`build_bit_mask`](@ref)
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
abstract type AbstractWindowsFilter{T<:AbstractFilterLimiter} <: AbstractFilterBased{T} end

# -----------------------------------------------------------------------------------------
# AbstractFeaturesSelector

"""
    apply(df, selector)

Return vector containing indicies of suitable attributes

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame to evaluate
- `selector::AbstractFeaturesSelector`: applied selector
"""
function apply(
    df::AbstractDataFrame,
    selector::AbstractFeaturesSelector{T}
)::Vector{Integer} where {T<:AbstractLimiter}
    return error("`apply` not implemented for type: " * string(typeof(selector)))
end

limiter(l::AbstractLimiter) = error("Not implemented for $(typeof(l))")
