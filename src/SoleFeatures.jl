module SoleFeatures

using StatsBase
using SoleTraits
using DynamicAxisWarping
using Reexport
using Random
using Catch22
using LinearAlgebra
using OrderedCollections

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
export MeasuresRanking
export WindowsFilter
# main functions
export apply, buildbitmask, transform, transform!
# utils
export bm2attr

@reexport using DataFrames
@reexport using SoleBase
@reexport using SoleTraits

# windows: should be moved
include("./windows/data-filters.jl")
include("./windows/windows.jl")
# general utils
include("./utils/utils.jl")
# limiters
include("./limiter/interfaces.jl")
include("./limiter/functions.jl")
# selectors
include("./interfaces.jl")
include("./functions.jl")
# variance
include("./variancefilter.jl")
# correlation
include("./correlationfilter/utils.jl")
include("./correlationfilter/correlationfilter.jl")
# random
include("./randomfilter.jl")
# measures
include("./measuresfilter.jl")
# windowsf
include("./windowsfilter/utils.jl")
include("./windowsfilter/windowsfilter.jl")

end # module
