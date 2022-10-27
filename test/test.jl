using SoleFeatures
using StatsBase

include("/home/patrik/develop/aclai/sole/SoleFeatures.jl/test/test_function.jl")

df = random_timeseries_df(;ninstances=5, nattr=3, ts_len=10)

limiter = SoleFeatures.GroupFittestLimiter(0.5)
vt = VarianceThreshold(0.02)
fmw = SoleFeatures.FixedNumMovingWindows(3, 0.25)
measures = [mean, minimum, maximum]
selector = WindowsFilter(limiter, vt, fmw, measures, :Attributes)

# tnd = @elapsed newdf = SoleFeatures.expand(df, selector);
newdf = SoleFeatures.expand(df, selector);
# println("New dataframe $(tnd):")
println("New dataframe:")
println(size(newdf))

# texp = @elapsed expansions = SoleFeatures.evaluate(df, selector);
expansions = SoleFeatures.evaluate(df, selector);
println("Expansions:")
println(length(expansions))
# tnd = @elapsed newdf = SoleFeatures.expand(df, expansions);
newdf = SoleFeatures.expand(df, expansions);
# println("Sliced new dataframe (evaluation time: $(texp), expansions time: $(texp):")
println("Sliced new dataframe:")
println(size(newdf))


# df = random_timeseries_df(;ninstances=5, nattr=10, ts_len=10)
# nbest = 5
# vr = VarianceRanking(nbest)

# mr = MeasuresRanking(nbest, vr)

# wr = WindowsFilter(
#     RankingLimiter(nbest, true),
#     vr,
#     SoleFeature.FixedNumMovingWindows(1, 0.0),
#     [ Measure(p) for p in pairs(SoleFeatures._MEASURES) ]
#     SoleFeature.GROUPBY_ATTRIBUTES
# )

# bmmr = buildbitmask(df, mr)
# wrbm = buildbitmask(df, wr)
