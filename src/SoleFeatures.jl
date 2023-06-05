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
export AbstractFilterBased
export AbstractWrapperBased
export AbstractEmbeddedBased
export AbstractLimiter
# structs
export VarianceThreshold
export VarianceRanking
export RandomRanking
export StatisticalAtLeastOnce
export StatisticalMajority
export CompoundStatisticalAtLeastOnce
export CompoundStatisticalMajority
export CorrelationFilter
# main functions
export apply, buildbitmask, transform, transform!
# utils
export bm2var

@reexport using DataFrames

include("interface.jl")
include("core.jl")
# Utils
include("utils/utils.jl")
# Filters
# include("filters/limiter.jl")
include("filters/interface.jl")
include("filters/univariate/randomfilter.jl")
include("filters/univariate/statisticalfilter.jl")
include("filters/univariate/variancefilter.jl")
include("filters/univariate/chi2filter.jl")
include("filters/univariate/utils.jl")
include("filters/multivariate/correlationfilter.jl")
# Experimental
include("experimental/Experimental.jl")
import .Experimental

end # module
