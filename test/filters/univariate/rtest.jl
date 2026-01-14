using Test
using SoleFeatures

using RDatasets

boston = dataset("MASS", "Boston")
Xr     = Matrix(boston[:, 1:end-1])
yr     = boston[:, end]

python_dir()  = joinpath(dirname(@__FILE__), "data")
pyt(filename) = joinpath(python_dir(), filename)

# ---------------------------------------------------------------------------- #
#                              r-test regression                               #
# ---------------------------------------------------------------------------- #
# from sklearn.datasets import fetch_openml
# from sklearn.feature_selection import r_regression

# boston = fetch_openml(name="boston", version=1)
# X = boston.data
# y = boston.target
# r_regress = r_regression(X, y)

pyt_file          = pyt("r_regress.txt")
sk_regress_result = readlines(pyt_file)
sk_regress_result = parse.(Float64, sk_regress_result)
    
r_regress_result  = RtestFilter(Xr, yr)
r_regress_score   = get_score(r_regress_result)

@test isapprox(r_regress_score, sk_regress_result)

@test eltype(r_regress_result)             == Float64
@test get_task(r_regress_result)           == RegressionTask
@test get_learning(r_regress_result)       == Supervised
@test get_dimensionality(r_regress_result) == Univariate
@test_nowarn get_rank(r_regress_result)
@test_nowarn get_score(r_regress_result)

# ---------------------------------------------------------------------------- #
#                                  float32                                     #
# ---------------------------------------------------------------------------- #
X32 = rand(Float32, 100, 10)
y32 = rand(Float32, 100)
filter32 = RtestFilter(X32, y32)
@test eltype(get_score(filter32)) == Float32