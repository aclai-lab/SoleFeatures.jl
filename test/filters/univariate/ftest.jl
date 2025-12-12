using Test
using SoleFeatures

using RDatasets

iris = dataset("datasets", "iris")
Xc   = Matrix(iris[:,1:end-1])
yc   = iris[:,end]

boston = dataset("MASS", "Boston")
Xr     = Matrix(boston[:, 1:end-1])
yr     = boston[:, end]

python_dir()  = joinpath(dirname(@__FILE__), "data")
pyt(filename) = joinpath(python_dir(), filename)

# ---------------------------------------------------------------------------- #
#                           f-test classification                              #
# ---------------------------------------------------------------------------- #
# from sklearn.feature_selection import f_classif
# from sklearn import datasets

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# f_statistic, p_values = f_classif(X, y)

pyt_file          = pyt("f_classif.txt")
sk_classif_result = readlines(pyt_file)
sk_classif_result = parse.(Float64, sk_classif_result)

f_classif_result  = FtestFilter(Xc, yc)
f_classif_score   = get_score(f_classif_result)

@test isapprox(f_classif_score, sk_classif_result)

# ---------------------------------------------------------------------------- #
#                              f-test regression                               #
# ---------------------------------------------------------------------------- #
# from sklearn.datasets import fetch_openml
# from sklearn.feature_selection import f_regression

# boston = fetch_openml(name="boston", version=1)
# X = boston.data
# y = boston.target
# f_regress, p_values = f_regression(X, y)

pyt_file          = pyt("f_regress.txt")
sk_regress_result = readlines(pyt_file)
sk_regress_result = parse.(Float64, sk_regress_result)
    
f_regress_result  = FtestFilter(Xr, yr)
f_regress_score   = get_score(f_regress_result)

@test isapprox(f_regress_score, sk_regress_result)

