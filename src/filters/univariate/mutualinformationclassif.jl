# ---------------------------------------------------------------------------- #
#                       mutual information classificator                       #
# ---------------------------------------------------------------------------- #
struct MutualInformationClassif{T <: AbstractLimiter} <: AbstractMutualInformationClassif{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMutualInformationClassif) = true
is_unsupervised(::AbstractMutualInformationClassif) = false

function score(X::AbstractMatrix, y::AbstractVector{<:Class}, selector::MutualInformationClassif)
    return mutual_info_classif(X, y)
end
score(Xdf::AbstractDataFrame, y::AbstractVector{<:Class}, selector::MutualInformationClassif) = score(Matrix(Xdf), y, selector)

# Ranking
MutualInformationClassifRanking(nbest) = MutualInformationClassif(RankingLimiter(nbest, true))
