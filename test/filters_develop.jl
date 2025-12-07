using SoleFeatures

using CSV
using DataFrames

ds_dir()    = joinpath(dirname(@__FILE__), "data/csv")
ds(filename) = joinpath(ds_dir(), filename)
ds_file = ds("winequality_complete.csv")
dataframe = DataFrame(CSV.File(ds_file))

X = dataframe[:, 2:end]
X = Matrix{Float64}(X)
y = dataframe[:, 1]
y = MLJ.levelcode.(categorical(y))

f_scores = mrmr_classif(X, y)
f_sorted_idxs = sortperm(scores, rev=true)

ks_scores = mrmr_classif(X, y; relevance=kolmogorov_smirnov)
ks_sorted_idxs = sortperm(scores, rev=true)

rf_scores = mrmr_classif(X, y; relevance=random_forest)
rf_sorted_idxs = sortperm(scores, rev=true)

#####################################################################################ù
using Statistics, StatsBase
T = Float64
relevance = f_statistic
# redundancy = Statistics.cor
redundancy = StatsBase.cor
denominator = mean

function select_feature!(
    scores::Vector{T},
    rel_result::Vector{T},
    rel_select::BitVector,
    score_denominator::Vector{T},
) where T
    score = rel_result[rel_select] ./ score_denominator[rel_select]
    idx = argmax(rel_result)
    rel_result[idx] = zero(T)
    rel_select[idx] = 0
    scores[idx] = maximum(score)
end

rel_result = relevance(X, y)
nfeats = length(rel_result)
scores = zeros(T, nfeats)
rel_select = BitVector(ones(Bool, nfeats))
score_denominator = ones(T, nfeats)

for _ in 1:nfeats
    select_feature!(scores, rel_result, rel_select, score_denominator)
    score_denominator[rel_select] = denominator(abs.(redundancy(X[:,rel_select], X[:,.!rel_select])), dims=2)
end

