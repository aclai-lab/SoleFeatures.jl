using SoleFeatures

include("./test_function.jl")

# X = random_df();
# y = rand(["S", "H", "M"], nrow(X))
# st = StatisticalThreshold(1)

# if (is_supervised(st))
#     println("Supervised selector using transform(X, y, sel)")
#     transform!(X, y, st)
#     println("Something works 😸")
#     println(X)
# else
#     println("Something goes wrong 😿")
# end

X = random_timeseries_df()
grplim = SoleFeatures.GroupFittestLimiter(0.5)
vr = VarianceThreshold(0.02)
fmw = SoleFeatures.FixedNumMovingWindows(3, 0.25)
m = [minimum, maximum]
wf = WindowsFilter(grplim, vr, fmw, m, :Attributes)

df1 = SoleFeatures.evaluate(X, wf)
df2 = SoleFeatures.evaluate2(X, wf)
