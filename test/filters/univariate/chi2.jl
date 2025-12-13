using Test
using SoleFeatures

using RDatasets

iris = dataset("datasets", "iris")
Xc   = Matrix(iris[:,1:end-1])
yc   = iris[:,end]

python_dir()  = joinpath(dirname(@__FILE__), "data")
pyt(filename) = joinpath(python_dir(), filename)

# ---------------------------------------------------------------------------- #
#                           f-test classification                              #
# ---------------------------------------------------------------------------- #
# from sklearn.feature_selection import chi2
# from sklearn import datasets

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# chi2, p_values = chi2(X, y)

pyt_file          = pyt("chi2.txt")
sk_classif_result = readlines(pyt_file)
sk_classif_result = parse.(Float64, sk_classif_result)

chi2_result  = Chi2Filter(Xc, yc)
chi2_score   = get_score(chi2_result)

@test isapprox(chi2_score, sk_classif_result)

@test eltype(chi2_result)             == Float64
@test get_task(chi2_result)           == ClassificationTask
@test get_learning(chi2_result)       == Supervised
@test get_dimensionality(chi2_result) == Univariate
@test_nowarn get_rank(chi2_result)
@test_nowarn get_score(chi2_result)


