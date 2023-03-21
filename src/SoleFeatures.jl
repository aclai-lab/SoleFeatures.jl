# __precompile__()

module SoleFeatures

using StatsBase
using SoleTraits
using DynamicAxisWarping
using Reexport
using Random
using Catch22
using LinearAlgebra
using OrderedCollections
using HypothesisTests
using IterTools
using PyCall

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
# export MeasuresRanking
# export WindowsFilter
export StatisticalThreshold
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
@reexport using SoleBase
@reexport using SoleTraits

# windows: should be moved
include("windows/windows.jl")
# limiters
include("limiters/interface.jl")
include("limiters/core.jl")
# general utils
include("utils/utils.jl")
# selectors
include("interface.jl")
include("core.jl")
## filters
### variance
include("filters/variancefilter.jl")
### correlation
include("filters/correlationfilter/utils.jl")
include("filters/correlationfilter/correlationfilter.jl")
### random
include("filters/randomfilter.jl")
### windowsf
include("filters/windowsfilter/utils.jl")
include("filters/windowsfilter/windowsfilter.jl")
## statisticals
include("filters/statisticalfilter.jl")

end # module
