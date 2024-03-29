# =========================================================================================
# abstract types

"""
Abstract supertype for all features selector.

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

# =========================================================================================
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
    selector::AbstractFeaturesSelector
)
    return error("Unsupervised `apply` not implemented for: $(typeof(selector))")
end

function apply(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::AbstractFeaturesSelector
)
    return error("Supervised `apply` not implemented for: $(typeof(selector))")
end
