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

include("./utils.jl")
include("./interfaces.jl")
include("./functions.jl")
# variance
include("./variance/variance_threshold.jl")
include("./variance/variance_ranking.jl")
# correlation
include("./correlation/utils.jl")
include("./correlation/correlation_threshold.jl")
include("./correlation/correlation_ranking.jl")
# random
include("./random_ranking.jl")
# measures
include("./measures_ranking.jl")

end # module
