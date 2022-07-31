# TODO: make comments in minmax_normalize
# TODO: utils.minmax_normalize shouldn't flat cols
# TODO: utils.minmax_normalize test dimension 2
# TODO: better implmentation of selector_function on correlation_ranking and correlation_threshold
# TODO: apply function have to return mfd and bit mask
# TODO: implement
# TODO: in function Int -> Integer

module SoleFeatures

using StatsBase
using DynamicAxisWarping
using Reexport

export VarianceThreshold
export VarianceRanking
export AbstractFeaturesSelector
export CorrelationRanking

@reexport using DataFrames
@reexport using SoleBase
@reexport using Revise

include("./utils.jl")
include("./interfaces.jl")
include("./variance_threshold.jl")
include("./variance_ranking.jl")
include("./correlation_commons.jl")
include("./correlation_threshold.jl")
include("./correlation_ranking.jl")

end # module
