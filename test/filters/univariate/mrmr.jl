using Test
using SoleFeatures

using MAT
using MLJ
using DataFrames

Xc, yc = @load_iris
Xc = Matrix(DataFrame(Xc))

scores = SoleFeatures.score(MutualInformationClassif(IdentityLimiter()), Xc, yc)