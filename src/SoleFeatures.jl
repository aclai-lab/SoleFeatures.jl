module SoleFeatures

using Tables
using StatsBase
using SoleData
using Reexport
using Random
using LinearAlgebra
using HypothesisTests
using IterTools
using PyCall
using MLBase

# abstracts
export AbstractFeatureSelector
export AbstractFilterBasedFS
export AbstractWrapperBasedFS
export AbstractEmbeddedBasedFS
export AbstractLimiter
# structs

# main functions
export select, apply, buildbitmask, transform, transform!
# utils
export bm2var

@reexport using DataFrames

include("interface.jl")
include("core.jl")
# Utils
include("utils/utils.jl")
# Filters
include("filters/interface.jl")
include("filters/univariate/randomcriterion.jl")
include("filters/univariate/statisticalcriterion.jl")
include("filters/univariate/variancecriterion.jl")
include("filters/univariate/chi2criterion.jl")
# include("filters/univariate/utils.jl")
# include("filters/multivariate/correlationfilter.jl")
# Experimental
include("experimental/Experimental.jl")
import .Experimental

end # module
