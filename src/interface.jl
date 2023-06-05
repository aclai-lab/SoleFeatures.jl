# =========================================================================================
# abstract types

const Class = Union{AbstractString,Symbol}
const Dataset = Union{AbstractDataFrame,AbstractMatrix}

"""
Abstract supertype for all features selector.

"""
abstract type AbstractFeatureSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractFilterBasedFS <: AbstractFeatureSelector end

select(::AbstractFilterBasedFS, X::Dataset, args...; kwargs...) = error("")
transform!(::AbstractFilterBasedFS, X::Dataset, args...; kwargs...) = error("")

"""
Abstract supertype filter based selector.
"""
abstract type AbstractWrapperBasedFS <: AbstractFeatureSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractEmbeddedBasedFS <: AbstractFeatureSelector end
