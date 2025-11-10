module SoleFeatures

using StatsBase
using MultiData
using SoleData
using Random
using LinearAlgebra
using HypothesisTests
using IterTools
using MLBase

using DataTreatments
using SparseArrays
using CategoricalArrays

using SpecialFunctions  # For digamma function
using NearestNeighbors  # For KDTree and knn

# ---------------------------------------------------------------------------- #
#                                  abstracts                                   #
# ---------------------------------------------------------------------------- #
export AbstractFeaturesSelector
export AbstractFilterBased
export AbstractWrapperBased
export AbstractEmbeddedBased
export AbstractLimiter

# ---------------------------------------------------------------------------- #
#                                    main                                      #
# ---------------------------------------------------------------------------- #
export AbstractFilterBased
include("interface.jl")

export apply, buildbitmask, transform, transform!
include("core.jl")

export bm2var
include("utils/utils.jl")
include("utils/normalize.jl")

export Score, GroupScore
include("selection/interface.jl")

# ---------------------------------------------------------------------------- #
#                                   filters                                    #
# ---------------------------------------------------------------------------- #
export PercentageLimiter
include("filters/limiter.jl")
include("filters/interface.jl")

export CompoundStatisticalAtLeastOnce, CompoundStatisticalMajority
include("filters/univariate/utils.jl")

export Chi2Ranking, Chi2Threshold
include("filters/univariate/chi2filter.jl")

export CorrelationFilter
include("filters/multivariate/correlationfilter.jl")

include("filters/univariate/fisherscorefilter.jl")

export IdentityFilter, IdentityLimiter
include("filters/univariate/identityfilter.jl")

include("filters/mutual_info.jl")

export MutualInformationClassif, MutualInformationClassifRanking
include("filters/univariate/mutualinformationclassif.jl")

export PearsonCorFilter, PearsonCorRanking
include("filters/univariate/pearsoncorfilter.jl")

export RandomRanking
include("filters/univariate/randomfilter.jl")

export StatisticalFilter, StatisticalLimiter
include("filters/univariate/statisticalfilter.jl")

include("filters/univariate/suplapscorefiler.jl")

export VarianceFilter, VarianceRanking, VarianceThreshold
include("filters/univariate/variancefilter.jl")

# ---------------------------------------------------------------------------- #
#                             feature selection                                #
# ---------------------------------------------------------------------------- #
export feature_selection
include("selection/fselection.jl")

end
