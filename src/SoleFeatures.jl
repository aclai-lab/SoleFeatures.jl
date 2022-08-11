# TODO: make comments in minmax_normalize
# TODO: better implmentation of selector_function on correlation_ranking and correlation_threshold

module SoleFeatures

using StatsBase
using DynamicAxisWarping
using Reexport
using Random

export AbstractFeaturesSelector
export VarianceThreshold
export VarianceRanking
export CorrelationRanking
export CorrelationThreshold
export RandomRanking
export build_bitmask, apply, apply!

@reexport using DataFrames
@reexport using SoleBase
@reexport using Revise

include("./utils.jl")
include("./interfaces.jl")
include("./functions.jl")
include("./variance_threshold.jl")
include("./variance_ranking.jl")
include("./correlation_commons.jl")
include("./correlation_threshold.jl")
include("./correlation_ranking.jl")
include("./random_ranking.jl")

end # module
