module SoleFeatures

using StatsBase
using SoleTraits
using DynamicAxisWarping
using Reexport
using Random
using Catch22
using LinearAlgebra
using OrderedCollections
using HypothesisTests
using IterTools
using PyCall

# abstracts
export AbstractFeaturesSelector
export AbstractFilterBased
export AbstractWrapperBased
export AbstractEmbeddedBased
export AbstractLimiter
export AbstractFilterLimiter
export AbstractWrapperLimiter
export AbstractEmbeddedLimiter
# structs
export VarianceThreshold
export VarianceRanking
export CorrelationRanking
export CorrelationThreshold
export RandomRanking
# export MeasuresRanking
# export WindowsFilter
export StatisticalThreshold
# main functions
export apply, buildbitmask, transform, transform!
# utils
export bm2attr

@reexport using DataFrames
@reexport using SoleBase
@reexport using SoleTraits

# windows: should be moved
include("windows/windows.jl")
# limiters
include("limiters/interfaces.jl")
include("limiters/functions.jl")
# general utils
include("utils/utils.jl")
# selectors
include("interfaces.jl")
include("functions.jl")
## variance
include("filters/variancefilter.jl")
## correlation
include("filters/correlationfilter/utils.jl")
include("filters/correlationfilter/correlationfilter.jl")
## random
include("filters/randomfilter.jl")
## measures
include("filters/measuresfilter.jl")
# windowsf
include("filters/windowsfilter/utils.jl")
include("filters/windowsfilter/windowsfilter.jl")
# statisticals
include("filters/statisticalfilter.jl")

end # module
