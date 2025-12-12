using Test
using SoleFeatures

using MLJ
using RDatasets

iris = dataset("datasets", "iris")
X = Matrix(iris[:,1:end-1])
y = iris[:,end]
y = MLJ.levelcode.(y)

python_dir()  = joinpath(dirname(@__FILE__), "data")
pyt(filename) = joinpath(python_dir(), filename)

# ---------------------------------------------------------------------------- #
#                           f-test classification                              #
# ---------------------------------------------------------------------------- #
# import sklearn
# from sklearn.feature_selection import f_classif
# from sklearn import datasets
# iris = datasets.load_iris()

# X = iris.data
# y = iris.target
# f_statistic, p_values = f_classif(X, y)

pyt_file          = pyt("f_classif.txt")
sk_classif_result = readlines(pyt_file)
sk_classif_result = parse.(Float64, sk_classif_result)

f_classif_result  = FtestFilter(X, y)
f_classif_score   = get_score(f_classif_result)

@test isapprox(f_classif_score, sk_classif_result)
