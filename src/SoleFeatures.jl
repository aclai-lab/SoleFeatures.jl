# TODO: make comments in minmax_normalize
# TODO: better implmentation of selector_function on correlation_ranking and correlation_threshold

module SoleFeatures

using StatsBase
using SoleTraits
using DynamicAxisWarping
using Reexport
using Random
using Catch22
using LinearAlgebra

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
# main functions
export apply, buildbitmask, transform, transform!
# utils
export bm2attr

@reexport using DataFrames
@reexport using SoleBase
@reexport using SoleTraits

# utils
include("./utils/utils.jl")
include("./utils/data-filters.jl")
# limiters
include("./limiter/interfaces.jl")
include("./limiter/functions.jl")
# selectors
include("./interfaces.jl")
include("./functions.jl")
# variance
include("./variancefilter.jl")
# correlation
include("./correlation/utils.jl")
include("./correlation/correlationfilter.jl")
# random
include("./randomfilter.jl")
# measures
include("./measuresfilter.jl")

end # module
