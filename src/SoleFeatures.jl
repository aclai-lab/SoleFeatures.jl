module SoleFeatures

using DataFrames
using SoleBase

# -----------------------------------------------------------------------------------------
# exports

export VarianceThreshold

# -----------------------------------------------------------------------------------------
# abstract types

"""
Abstract supertype for all features selector.

A concrete subtype of AbstractFeaturesSelector should always provide functions
[`apply`](@ref) and [`build_bit_mask`](@ref)
"""
abstract type AbstractFeaturesSelector end

"""
Abstract supertype for all univariate features selector.
"""
abstract type AbstractUnivariateSelector <: AbstractFeaturesSelector end

"""
Abstract supertype for all multivariate features selector.
"""
abstract type AbstractMultivariateSelector <: AbstractFeaturesSelector end

# -----------------------------------------------------------------------------------------
# AbstractFeaturesSelector

"""
    apply(mfd, selector)

Return a new MultiFrameDataset from `mfd` without the attributes considered unsitable from `selector`
## ARGUMENTS
- `mfd::AbstractMultiFrameDataset`: AbstracttMultiFrameDataset on which apply the selector
- `selector::AbstractFeaturesSelector`: applied selector
"""
function apply(mfd::SoleBase.AbstractMultiFrameDataset, selector::AbstractFeaturesSelector)
    return error("`apply` not implmented for type: "
        * string(typeof(selector)))
end

"""
    apply!(mfd, selector)

Remove form `mfd` attributes considered unsitable from `selector`

## ARGUMENTS
- `mfd::AbstractMultiFrameDataset`: AbstracttMultiFrameDataset on which apply the selector
- `selector::AbstractFeaturesSelector`: applied selector
"""
function apply!(mfd::SoleBase.AbstractMultiFrameDataset, selector::AbstractFeaturesSelector)
    return error("`apply` not implmented for type: "
        * string(typeof(selector)))
end

"""
    build_bit_mask(df, selector)

Return a bit vector representing which attributes in `df` are considered suitable or not by the `selector`
(1 suitable, 0 not suitable)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame to evaluate
- `selector::AbstractFeaturesSelector`: applied selector
"""
function build_bit_mask(df::AbstractDataFrame, selector::AbstractFeaturesSelector)::BitVector
    return error("`build_bit_mask` not implmented for type: "
        * string(typeof(selector)))
end

# -----------------------------------------------------------------------------------------
# AbstractUnivariateSelector

function selector_threshold(selector::AbstractUnivariateSelector)
    return error("`selector_threshold` not implmented for type: "
        * string(typeof(selector)))
end

function selector_function(selector::AbstractUnivariateSelector)
    return error("`selector_function` not implmented for type: "
        * string(typeof(selector)))
end

include("./VarianceThreshold.jl")

end # module
