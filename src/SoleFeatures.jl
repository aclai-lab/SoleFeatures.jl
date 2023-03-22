module SoleFeatures

using StatsBase
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

# windows: should be moved
include("windows/windows.jl")
# limiters
include("limiters/interface.jl")
include("limiters/core.jl")
# general utils
include("utils/utils.jl")
# selectors
include("interface.jl")
include("core.jl")
## filters
### variance
include("filters/variancefilter.jl")
### correlation
include("filters/correlationfilter/utils.jl")
include("filters/correlationfilter/correlationfilter.jl")
### random
include("filters/randomfilter.jl")
### windowsf
include("filters/windowsfilter/utils.jl")
include("filters/windowsfilter/windowsfilter.jl")
## statisticals
include("filters/statisticalfilter.jl")

end # module
