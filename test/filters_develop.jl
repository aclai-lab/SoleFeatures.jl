using SoleFeatures

using MLJ
using CSV
using DataFrames
using StatsBase

ds_dir()    = joinpath(dirname(@__FILE__), "data/csv")
ds(filename) = joinpath(ds_dir(), filename)
ds_file = ds("winequality_complete.csv")
dataframe = DataFrame(CSV.File(ds_file))

X = dataframe[:, 2:end]
Xm = Matrix{Float64}(X)
y = dataframe[:, 1]
y = MLJ.levelcode.(categorical(y))

ftest = FtestFilter(Xm, y)

Xc, yc = @load_iris
Xc = DataFrame(Xc)
Xc = Matrix(Xc)

# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
struct PearsonCorrInfo
    redundancy  :: Base.Callable
    denominator :: Base.Callable
end

struct PearsonCorrFilter
    rank  :: Vector{Int64}
    score :: Vector{Real}
    info  :: PearsonCorrInfo

    function PearsonCorrFilter(
        X           :: AbstractArray,
        unifilt     :: AbstractFilterBased{F,T,L,D};
        redundancy  :: Base.Callable=StatsBase.cor,
        denominator :: Base.Callable=mean   
    ) where {F<:Real,T,L,D<:Univariate}
        uniscore = get_score(unifilt)
        rank, score = _pearson_corr(X, uniscore, StatsBase.cor, mean)
        info = PearsonCorrInfo(redundancy, denominator)
        new(rank, score, info)
    end
end

# ---------------------------------------------------------------------------- #
#              minimum redundancy maximum relevance classifier                 #
# ---------------------------------------------------------------------------- #
function _pearson_corr(
    X           :: AbstractArray,
    uniscore    :: AbstractVector{T},
    redundancy  :: Base.Callable,
    denominator :: Base.Callable
) where {T<:Real}
    function select_feature!(
        scores::Vector{T},
        uniscore::Vector{T},
        rel_select::BitVector,
        score_denominator::Vector{T},
    ) where T
        scores[rel_select] = uniscore[rel_select] ./ score_denominator[rel_select]
        active_idxs = findall(rel_select)
        idx_in_active = argmax(scores[rel_select])
        idx = active_idxs[idx_in_active]
        rel_select[idx] = 0
        push!(result, idx)
    end

    nfeats = length(uniscore)
    scores = zeros(T, nfeats)
    rel_select = BitVector(ones(Bool, nfeats))
    score_denominator = ones(T, nfeats)
    result = Int64[]

    for _ in 1:nfeats
        select_feature!(scores, uniscore, rel_select, score_denominator)
        score_denominator[rel_select] = denominator(abs.(redundancy(X[:,rel_select], X[:,.!rel_select])), dims=2)
    end

    return result
end

X = Xm
uniscore = Float64.(ftest.score)
redundancy = StatsBase.cor
denominator = mean
T = Float64

ftest = FtestFilter(Xc, yc)
a= _pearson_corr(Xc, ftest.score, StatsBase.cor, mean)


# 0.3636    0.2878    1.0025    0.6634