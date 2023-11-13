# =========================================================================================
# abstract types

# TODO add SoleBase.AbstractDataset or SoleData.AbstractMultiModalDataset to AnyDataset?
const AnyDataset = Union{AbstractDataFrame,AbstractMatrix}
const Class = Union{AbstractString,Symbol}

"""
Abstract supertype for all feature selector.s

"""
abstract type AbstractFeatureSelector end

function select(s::AbstractFeatureSelector, X::AnyDataset, args...; kwargs...)
    return error("Please, provide method " *
                 "select(::$(typeof(s)), X::$(typeof(X)), " *
                 "args...::$(typeof(args)); kwargs...::$(typeof(kwargs))).")

# =========================================================================================
# AbstractFeatureSelector

"""
    apply(X, selector)
    apply(X, y, selector)

Return vector containing indicies of suitable variables from selector.

## ARGUMENTS
- `X::AbstractDataFrame`: samples to evaluate
- `y::AbstractVector{<:Class}`: target vector
- `selector::AbstractFeatureSelector`: applied selector
"""
function apply(
    X::AbstractDataFrame,
    selector::AbstractFeatureSelector
)
    return error("Unsupervised `apply` not implemented for: $(typeof(selector))")
end

"""
Abstract supertype for filter based selectors.
"""
abstract type AbstractFilterBasedFS <: AbstractFeatureSelector end

"""
Abstract supertype for wrapper selectors.
"""
abstract type AbstractWrapperFS <: AbstractFeatureSelector end

"""
Abstract supertype for embedded selectors.
"""
abstract type AbstractEmbeddedFS <: AbstractFeatureSelector end
