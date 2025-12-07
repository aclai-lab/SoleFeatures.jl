using SoleFeatures

using MLJ
using CSV
using DataFrames

ds_dir()    = joinpath(dirname(@__FILE__), "data/csv")
ds(filename) = joinpath(ds_dir(), filename)
ds_file = ds("winequality_complete.csv")
dataframe = DataFrame(CSV.File(ds_file))

X = dataframe[:, 2:end]
Xm = Matrix{Float64}(X)
y = dataframe[:, 1]
y = MLJ.levelcode.(categorical(y))

f_result = mrmr_classif(Xm, y)
f_result = names(X)[f_result]

ks_result = mrmr_classif(Xm, y; relevance=kolmogorov_smirnov)
ks_result = names(X)[ks_result]

rf_result = mrmr_classif(Xm, y; relevance=random_forest)
rf_result = names(X)[rf_result]

#####################################################################################ù
using Statistics, StatsBase
T = Float64
# relevance = f_statistic
relevance = kolmogorov_smirnov
redundancy = StatsBase.cor
denominator = mean

function select_feature!(
    scores::Vector{T},
    rel_result::Vector{T},
    rel_select::BitVector,
    score_denominator::Vector{T},
) where T
    scores[rel_select] = rel_result[rel_select] ./ score_denominator[rel_select]
    active_idxs = findall(rel_select)
    idx_in_active = argmax(scores[rel_select])
    idx = active_idxs[idx_in_active]
    rel_select[idx] = 0
    push!(result, idx)
end

rel_result = relevance(Xm, y)
nfeats = length(rel_result)
scores = zeros(T, nfeats)
rel_select = BitVector(ones(Bool, nfeats))
score_denominator = ones(T, nfeats)
result = Int64[]

for _ in 1:nfeats
    select_feature!(scores, rel_result, rel_select, score_denominator)
    score_denominator[rel_select] = denominator(abs.(redundancy(Xm[:,rel_select], Xm[:,.!rel_select])), dims=2)
end

ks2 = names(X)[result]

#################################################################################
T=Float64

class_sum = sums_args
class_counts = n_samples
grand_sqr = square_of_sums_alldata
class_sqr = square_of_sums_args

classes = unique(y)
nclasses, n = length(classes), length(x)

class_sqr    = Vector{T}(undef, nclasses)
@inbounds for (i, c) in enumerate(classes)
    mask = y .== c
    class_sqr[i]    = sum(@view x[mask])^2 / sum(mask)
end

grand_sqr  = sum(x)^2 / n

ss_between = sum(class_sqr[i] for i in 1:nclasses) - grand_sqr
ss_within  = sum(x.^2) - grand_sqr - ss_between

ms_between = ss_between / (nclasses - 1)
ms_within = ss_within / (n - nclasses)

return iszero(ms_within) ? zero(T) : ms_between / ms_within