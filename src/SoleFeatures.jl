module SoleFeatures

using StatsBase
using MultiData
using SoleData
using Random
using LinearAlgebra
using HypothesisTests
using IterTools
using MLBase
using Distributions

using DataTreatments
using SparseArrays
using CategoricalArrays

using NearestNeighbors  # used by mutual information classifier filter

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

export Score, GroupScore
include("selection/interface.jl")

# ---------------------------------------------------------------------------- #
#                                   filters                                    #
# ---------------------------------------------------------------------------- #
export IdentityLimiter, ThresholdLimiter, RankingLimiter
export MajorityLimiter, AtLeastLimiter, PercentageLimiter
include("filters/limiter.jl")
include("filters/interface.jl")

export CompoundStatisticalAtLeastOnce, CompoundStatisticalMajority
include("filters/univariate/utils.jl")

export Chi2Filter, chi2
export Chi2Ranking, Chi2Threshold
include("filters/univariate/chi2.jl")

export FisherScoreFilter, fisher_score
export FisherScoreRanking, FisherScoreThreshold
include("filters/univariate/fisherscore.jl")

export IdentityFilter
include("filters/univariate/identityfilter.jl")

export MutualInformationClassif, mutual_info_classifier
export MutualInformationClassifRanking, MutualInformationClassifThreshold
include("filters/univariate/mutualinformationclassif.jl")

export CorrelationFilter
include("filters/multivariate/correlationfilter.jl")

include("filters/mutual_info.jl")

export PearsonCorFilter
export get_pearson_cor_identity, get_pearson_cor_threshold
export get_pearson_cor_ranking, get_pearson_cor_percentage
include("filters/univariate/pearsoncorfilter.jl")

export RandomFilter
export get_random_identity, get_random_threshold
export get_random_ranking, get_random_percentage
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
