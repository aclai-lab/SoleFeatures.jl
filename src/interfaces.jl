# -----------------------------------------------------------------------------------------
# abstract types

"""
Abstract supertype for all features selector.

A concrete subtype of AbstractFeaturesSelector should always provide functions
[`transform`](@ref) and [`build_bit_mask`](@ref)
"""
abstract type AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractFilterBased <: AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractWrapperBased <: AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractEmbeddedBased <: AbstractFeaturesSelector end

# -----------------------------------------------------------------------------------------
# Correlation

abstract type AbstractCorrelationFilter <: AbstractFilterBased end

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
    selector::AbstractFeaturesSelector
)::Vector{Integer}
    return error("`apply` not implmented for type: " * string(typeof(selector)))
end
