using SoleFeatures
using Test
using Sole
using Random

# ---------------------------------------------------------------------------- #
#                             dataset preprocess                               #
# ---------------------------------------------------------------------------- #
# load a time-series dataset
X, y       = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# prepare dataset for feature selection
features = [mean, std]
nwindows = 5
X_features = @test_nowarn SoleFeatures.feature_selection_preprocess(X; features, nwindows)

# # ---------------------------------------------------------------------------- #
# #                       unsupervised feature selection                         #
# # ---------------------------------------------------------------------------- #
# # prepare selector for each group: grouping for (Variables, Measures) it will be 5 items (nwindows) in each group
# selector = VarianceRanking(nwindows)
# @test selector.limiter.nbest == 5

# # prepare group by: in this case it will be 56 groups (Variables * Measures)
# # const GROUPBY_VARIABLES = :Variables
# # const GROUPBY_WINDOWS = :Windows
# # const GROUPBY_MEASURES = :Measures
# groupbykey = [(SoleFeatures.GROUPBY_VARIABLES, SoleFeatures.GROUPBY_MEASURES)]

# # ================== PREPARE VARIABLES, WINDOWS, MEASURES ==================
# @info "PREPARE VARIABLES, WINDOWS, MEASURES"

# # prepare awmds
# vars = Symbol.(names(X))

# # struct FixedNumMovingWindows
# #     nwindows::Int
# #     reloverlap::Float64

# #     function FixedNumMovingWindows(nwindows::Integer, reloverlap::AbstractFloat)
# #         nwindows <= 0 && throw(DomainError(nwindows, "Must be greater than 0"))
# #         return new(nwindows, reloverlap)
# #     end
# # end
# # nwindows(mw::FixedNumMovingWindows) = mw.nwindows
# # function Base.length(mw::FixedNumMovingWindows)
# #     return nwindows(mw)
# # end

# # fnmw = FixedNumMovingWindows(3, 0.25)
# measures = [catch22..., minimum, maximum, StatsBase.mean]
# awmds = SoleFeatures.build_awmds(vars, [ fnmw... ], measures);

# # ================== UTILS ==================
# @info "UTILS"

# lenvars = length(vars)
# lenwins = nwindows
# lenmeasures = length(measures)
# lentot = lenvars * lenwins * lenmeasures
# println("# Variables: $(lenvars)")
# println("# Windows: $(lenwins)")
# println("# Measures: $(lenmeasures)")
# println("# Total features: $(lentot)")