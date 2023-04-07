# __precompile__()

module SoleFeatures

using StatsBase
using SoleData
using DynamicAxisWarping
using Reexport
using Random
using Catch22
using LinearAlgebra
using HypothesisTests
using IterTools
using PyCall

# abstracts
export AbstractFeaturesSelector
export AbstractFilterBased
export AbstractWrapperBased
export AbstractEmbeddedBased
export AbstractLimiter
# structs
export VarianceThreshold
export VarianceRanking
export RandomRanking
export StatisticalAtLeastOnce
export StatisticalMajority
export CompoundStatisticalAtLeastOnce
export CompoundStatisticalMajority
export CorrelationFilter
# main functions
export apply, buildbitmask, transform, transform!
# utils
export bm2attr

# consts
# const construct_w = PyNULL();
# const lap_score = PyNULL();
# const fisher_score = PyNULL();

# function __init__()

#     # init python packages
#     !PyCall.Conda.pip_interop(PyCall.Conda.ROOTENV) &&
#         PyCall.Conda.pip_interop(true, PyCall.Conda.ROOTENV) # allows environment to interact with pip
#     isempty(PyCall.Conda.parseconda(`list scipy`, PyCall.Conda.ROOTENV)) &&
#         PyCall.Conda.add("scipy", PyCall.Conda.ROOTENV)
#     isempty(PyCall.Conda.parseconda(`list scikit-learn`, PyCall.Conda.ROOTENV)) &&
#         PyCall.Conda.add("scikit-learn", PyCall.Conda.ROOTENV)
#     isempty(PyCall.Conda.parseconda(`list skfeature`, PyCall.Conda.ROOTENV)) &&
#         PyCall.Conda.pip("install", "git+https://github.com/jundongl/scikit-feature.git#egg=skfeature", PyCall.Conda.ROOTENV)

#     copy!(construct_w, pyimport_conda("skfeature.utility.construct_W.construct_W", "skfeature"))
#     copy!(lap_score, pyimport_conda("skfeature.function.similarity_based.lap_score", "skfeature"))
#     copy!(fisher_score, pyimport_conda("skfeature.function.similarity_based.fisher_score", "skfeature"))

# end

@reexport using DataFrames

include("interface.jl")
include("core.jl")
# Filters
include("filters/limiter.jl")
include("filters/interface.jl")
include("filters/univariate/randomfilter.jl")
include("filters/univariate/statisticalfilter.jl")
include("filters/univariate/variancefilter.jl")
include("filters/univariate/utils.jl")
include("filters/multivariate/correlationfilter.jl")
# Utils
include("utils/utils.jl")
# Experimental
include("experimental/Experimental.jl")
import .Experimental

end # module
