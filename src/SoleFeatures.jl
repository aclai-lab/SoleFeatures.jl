__precompile__()
module SoleFeatures

using StatsBase
using MultiData
using SoleData
using Reexport
using Random
using LinearAlgebra
using HypothesisTests
using IterTools
using PyCall
using MLBase
# using Pkg

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
export PearsonCorRanking
export Chi2Ranking
export Chi2Threshold
export MutualInformationClassifRanking
export CompoundStatisticalAtLeastOnce
export CompoundStatisticalMajority
export CorrelationFilter
# main functions
export apply, buildbitmask, transform, transform!
# utils
export bm2var

@reexport using DataFrames

const req_py_pkgs = ["scipy", "scikit-learn", "skfeature"]
const fs = PyNULL()
const construct_w = PyNULL()
const lap_score = PyNULL()
const fisher_score = PyNULL()
function __init__()

    pypkgs = getindex.(PyCall.Conda.parseconda(`list`, PyCall.Conda.ROOTENV), "name")
    needinstall = !all(p -> in(p, pypkgs), req_py_pkgs)

    if (needinstall)
        PyCall.Conda.pip_interop(true, PyCall.Conda.ROOTENV)
        PyCall.Conda.add("scipy")
        PyCall.Conda.add("scikit-learn")
        PyCall.Conda.pip(
            "install",
            "git+https://github.com/jundongl/scikit-feature.git#egg=skfeature",
            PyCall.Conda.ROOTENV
        )
    end

    copy!(fs, pyimport_conda("sklearn.feature_selection", "scikit-learn"))
    copy!(construct_w, pyimport_conda("skfeature.utility.construct_W", "skfeature"))
    copy!(lap_score, pyimport_conda(
        "skfeature.function.similarity_based.lap_score",
        "skfeature"
    ))
    copy!(fisher_score, pyimport_conda(
        "skfeature.function.similarity_based.fisher_score",
        "skfeature"
    ))
end

include("interface.jl")
include("core.jl")
# Utils
include("utils/utils.jl")
# Filters
include("filters/limiter.jl")
include("filters/interface.jl")
include("filters/univariate/randomfilter.jl")
include("filters/univariate/statisticalfilter.jl")
include("filters/univariate/variancefilter.jl")
include("filters/univariate/chi2filter.jl")
include("filters/univariate/pearsoncorfilter.jl")
include("filters/univariate/mutualinformationclassif.jl")
include("filters/univariate/suplapscorefiler.jl")
include("filters/univariate/fisherscorefilter.jl")
include("filters/univariate/utils.jl")
include("filters/multivariate/correlationfilter.jl")
# Experimental
include("experimental/Experimental.jl")
import .Experimental

end # module
