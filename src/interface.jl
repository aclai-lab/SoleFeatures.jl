# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
Abstract supertype for all features selector.
"""
abstract type AbstractFeaturesSelector end

is_supervised(::AbstractFeaturesSelector) = true
is_unsupervised(::AbstractFeaturesSelector) = true

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

# ---------------------------------------------------------------------------- #
#                                    types                                     #
# ---------------------------------------------------------------------------- #
const Class = Union{CategoricalValue, String, Symbol, Real}

# ---------------------------------------------------------------------------- #
#                            functions definitions                             #
# ---------------------------------------------------------------------------- #
"""
    apply(X, selector)
    apply(X, y, selector)

Return vector containing indicies of suitable variables from selector.

## ARGUMENTS
- `X::AbstractDataFrame`: samples to evaluate
- `y::AbstractVector{<:Class}`: target vector
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
    y::AbstractVector{<:Class},
    selector::AbstractFeaturesSelector
)
    return error("Supervised `apply` not implemented for: $(typeof(selector))")
end
