# -----------------------------------------------------------------------------------------
# abstract types

"""
Abstract supertype for all features selector.

A concrete subtype of AbstractFeaturesSelector should always provide functions
[`apply`](@ref) and [`build_bit_mask`](@ref)
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
# AbstractFeaturesSelector

"""
    build_bit_mask(df, selector)

Return a bit vector representing which attributes in `df` are considered suitable or not by the `selector`
(1 suitable, 0 not suitable)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame to evaluate
- `selector::AbstractFeaturesSelector`: applied selector
"""
function build_bitmask(
    df::AbstractDataFrame,
    selector::AbstractFeaturesSelector
)::BitVector
    return error("`build_bitmask` not implmented for type: " * string(typeof(selector)))
end
