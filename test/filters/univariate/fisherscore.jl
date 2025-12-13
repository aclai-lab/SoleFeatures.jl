using Test
using SoleFeatures

using RDatasets

iris = dataset("datasets", "iris")
Xc   = Matrix(iris[:,1:end-1])
yc   = iris[:,end]

# ---------------------------------------------------------------------------- #
#                        fisher score classification                           #
# ---------------------------------------------------------------------------- #

fisher_result  = FisherScoreFilter(Xc, yc)
fisher_score   = get_score(fisher_result)

@test eltype(fisher_result)             == Float64
@test get_task(fisher_result)           == ClassificationTask
@test get_learning(fisher_result)       == Supervised
@test get_dimensionality(fisher_result) == Univariate
@test_nowarn get_rank(fisher_result)
@test_nowarn get_score(fisher_result)

# ---------------------------------------------------------------------------- #
#                                  float32                                     #
# ---------------------------------------------------------------------------- #
X32 = rand(Float32, 100, 10)
y = rand(1:3, 100)
filter32 = FisherScoreFilter(X32, y)
@test eltype(get_score(filter32)) == Float32
