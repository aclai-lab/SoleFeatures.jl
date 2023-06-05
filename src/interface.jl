# =========================================================================================
# abstract types

const Class = Union{AbstractString,Symbol}
const Dataset = Union{AbstractDataFrame,AbstractMatrix}

"""
Abstract supertype for all features selector.

"""
abstract type AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractFilterBasedFS <: AbstractFeaturesSelector end

select(::AbstractFilterBasedFS, X::Dataset, args...; kwargs...) = error("")
transform!(::AbstractFilterBasedFS, X::Dataset, args...; kwargs...) = error("")

"""
Abstract supertype filter based selector.
"""
abstract type AbstractWrapperBasedFS <: AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractEmbeddedBasedFS <: AbstractFeaturesSelector end
