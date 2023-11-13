using DataFrames
using OrderedCollections
using StatsBase
using Revise
using Serialization
using SoleFeatures
using Catch22

# include("/home/patrik/develop/aclai/features_selection/results-sole/src/arff_2_mfd.jl")

isdefined(Main, :Catch22) && (Base.nameof(f::SuperFeature) = getname(f)) # wrap for Catch22

# ================== PREPARE DATASET ==================
@info "PREPARE DATASET"

# 28 variables 2 classes
# X, y = arff_2_mfd_multivariate("/home/patrik/develop/aclai/features_selection/results-sole/datasets/FingerMovements_TRAIN.arff")
# X = SoleData.modality(X, 1)

# ================== PREPARE VARIABLES, WINDOWS, MEASURES ==================
@info "PREPARE VARIABLES, WINDOWS, MEASURES"

# prepare awmds
vars = Symbol.(names(X))
fnmw = SoleFeatures.FixedNumMovingWindows(3, 0.25)
measures = [catch22..., minimum, maximum, StatsBase.mean]
awmds = SoleFeatures.build_awmds(vars, [ fnmw... ], measures);

# ================== UTILS ==================
@info "UTILS"

lenvars = length(vars)
lenwins = length(fnmw)
lenmeasures = length(measures)
lentot = lenvars * lenwins * lenmeasures
println("# Variables: $(lenvars)")
println("# Windows: $(lenwins)")
println("# Measures: $(lenmeasures)")
println("# Total features: $(lentot)")

# ================== STEP 1: UNSUPERVISED FEATURE SELECTION ==================
@info "STEP 1: UNSUPERVISED FEATURE SELECTION"

# prepare selector for each group: grouping for (Variables, Measures) it will be 3 item (windows) in each group
selector = VarianceRanking(lenwins)

# prepare group by: in this case it will be 56 groups (Variables * Measures)
groupbykey = [(SoleFeatures.GROUPBY_VARIABLES, SoleFeatures.GROUPBY_MEASURES)]

# prepare aggragate function to apply for each group
aggregatef = StatsBase.mean

# prepare limiter to retrive lentot/2 groups
limiter = SoleFeatures.RankingLimiter(Int(ceil(lenvars*lenmeasures/2)), true)

# prepare norm function
normf(X) = SoleFeatures.minmax_normalize(X; min_quantile=0.01, max_quantile=0.99, col_quantile=false) # TODO: change "col_quantile" in "mode" in something that accept symbol (:ALLVARIABLES, :BYVARIABLES)

# get result
awmds_res = SoleFeatures.evaluate(X, y, awmds, selector, groupbykey, aggregatef, limiter; normf=normf)
println("Length: $(length(awmds_res)/lenwins)")

# ================== STEP 2: SUPERVISED FEATURE SELECTION ==================
@info "STEP 2: SUPERVISED FEATURE SELECTION"

limiter = SoleFeatures.RankingLimiter(10, true)
awmds_res = SoleFeatures.evaluate(X, y, awmds_res, selector, groupbykey, aggregatef, limiter; normf=normf, supervised=true)
println("Length: $(length(awmds_res)/lenwins)")

# ================== STEP 3: VALIDATION ==================
@info "STEP 3: VALIDATION"

# It is sufficient that two population split correctly
statisticsfs = StatisticalThreshold(1)
limiter = SoleFeatures.ThresholdLimiter(0.01, >=) # TODO: use eps instead of 0.01?
awmds_check = SoleFeatures.evaluate(X, y, awmds_res, statisticsfs, groupbykey, aggregatef, limiter; normf=normf, supervised=true)

# ================== OUTPUT ==================

println("Tuple retrived from variance ranking: ")
println(join(unique!(join.(deleteat!.(split.(SoleFeatures._awm2str.(awmds_res), "@@"), 2), "@@")), '\n'))
println(length(awmds_res)/lenwins)

println("Validation:")
println(join(unique!(join.(deleteat!.(split.(SoleFeatures._awm2str.(awmds_check), "@@"), 2), "@@")), '\n'))
println(length(awmds_check)/lenwins)
