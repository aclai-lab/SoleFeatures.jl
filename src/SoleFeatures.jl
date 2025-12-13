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
using Statistics

using DataTreatments
using SparseArrays
using CategoricalArrays

using NearestNeighbors  # used by mutual information classifier filter
using DecisionTree      # used by mrmr filter

# ---------------------------------------------------------------------------- #
#                                   filter                                     #
# ---------------------------------------------------------------------------- #
export AbstractFilter
export AbstractTask, AbstractLearning, AbstractDimensionality
export ClassificationTask, RegressionTask
export Supervised, Unsupervised
export Univariate, Multivariate
export get_task, get_learning, get_dimensionality
export get_rank, get_score
include("filters/interface.jl")

export RtestFilter
include("filters/univariate/rtest.jl")

export FtestFilter
include("filters/univariate/ftest.jl")

export Chi2Filter
include("filters/univariate/chi2.jl")

# export AbstractFeaturesSelector
# export AbstractFilterBased
# export AbstractWrapperBased
# export AbstractEmbeddedBased
# export AbstractLimiter

# ---------------------------------------------------------------------------- #
#                                    main                                      #
# ---------------------------------------------------------------------------- #
# export AbstractFilterBased
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
# include("filters/interface.jl")

# export CompoundStatisticalAtLeastOnce, CompoundStatisticalMajority
# include("filters/univariate/utils.jl")

# export FisherScoreFilter, fisher_score
# export FisherScoreRanking, FisherScoreThreshold
# include("filters/univariate/fisherscore.jl")

export IdentityFilter
include("filters/univariate/identityfilter.jl")

# export KSTestFilter
# include("filters/univariate/ks-test.jl")

# export RandomForestFilter
# include("filters/univariate/random_forest.jl")

# export MrMrFilter, mrmr_classif
# export f_statistic, kolmogorov_smirnov, random_forest
# include("filters/univariate/mrmr.jl")

# export MutualInformationClassif, mutual_info_classifier
# export MutualInformationClassifRanking, MutualInformationClassifThreshold
# include("filters/univariate/mutualinformationclassif.jl")

# export CorrelationFilter
# include("filters/multivariate/correlationfilter.jl")

# include("filters/mutualinfo.jl")

# export PearsonCorFilter
# export get_pearson_cor_identity, get_pearson_cor_threshold
# export get_pearson_cor_ranking, get_pearson_cor_percentage
# include("filters/univariate/pearsoncorfilter.jl")

# export RandomFilter
# export get_random_identity, get_random_threshold
# export get_random_ranking, get_random_percentage
# include("filters/univariate/randomfilter.jl")

# export StatisticalFilter, StatisticalLimiter
# include("filters/univariate/statisticalfilter.jl")

# include("filters/univariate/suplapscorefiler.jl")

# export VarianceFilter
# export get_variance_identity, get_variance_threshold
# export get_variance_ranking, get_variance_percentage
# include("filters/univariate/variancefilter.jl")

# ---------------------------------------------------------------------------- #
#                             feature selection                                #
# ---------------------------------------------------------------------------- #
export feature_selection
include("selection/fselection.jl")

end
