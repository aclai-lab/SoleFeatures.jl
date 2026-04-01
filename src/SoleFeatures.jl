module SoleFeatures

using StatsBase
using MultiData
using SoleData
using Random
using LinearAlgebra
using IterTools
using MLBase

using DataTreatments
using SparseArrays
using CategoricalArrays

using Distributions     # used by Chi2Filter
using NearestNeighbors  # used by MutualInfoFilter
using DecisionTree      # used by RandomForestFilter
using HypothesisTests
using Statistics        # used by 

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
export AbstractFilter
export AbstractTask, AbstractLearning, AbstractDimensionality

export ClassificationTask, RegressionTask
export Supervised, Unsupervised
export Univariate, Multivariate

export get_task, get_learning, get_dimensionality
export get_rank, get_score
include("filters/interface.jl")

export Chi2Filter
include("filters/univariate/chi2.jl")

export FisherScoreFilter
include("filters/univariate/fisherscore.jl")

export FtestFilter
include("filters/univariate/ftest.jl")

export KSTestFilter
include("filters/univariate/kstest.jl")

export MutualInfoFilter
include("filters/univariate/mutualinfo.jl")

export RandomForestFilter
include("filters/univariate/random_forest.jl")

export RtestFilter
include("filters/univariate/rtest.jl")

export VarianceFilter
include("filters/univariate/variance.jl")


# export AbstractFeaturesSelector
# export AbstractFilterBased
# export AbstractWrapperBased
# export AbstractEmbeddedBased
# export AbstractLimiter

# ---------------------------------------------------------------------------- #
#                                   limiters                                   #
# ---------------------------------------------------------------------------- #
export AbstractLimiter

export ThresholdLimiter, RankingLimiter
export MajorityLimiter, AtLeastLimiter, PercentageLimiter
include("filters/limiters.jl")

# ---------------------------------------------------------------------------- #
#                             feature selection                                #
# ---------------------------------------------------------------------------- #
export feature_selection
include("selection/fselection.jl")

end
