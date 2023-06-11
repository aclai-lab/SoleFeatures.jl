# =========================================================================================
# abstract types

const Dataset = Union{AbstractDataFrame,AbstractMatrix}
const Class = Union{AbstractString,Symbol}

"""
Abstract supertype for all features selector.

"""
abstract type AbstractFeatureSelector end

select(::AbstractFeatureSelector, X::Dataset, args...; kwargs...) = error("")

"""
Abstract supertype filter based selector.
"""
abstract type AbstractFilterBasedFS <: AbstractFeatureSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractWrapperBasedFS <: AbstractFeatureSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractEmbeddedBasedFS <: AbstractFeatureSelector end
