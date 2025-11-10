# ---------------------------------------------------------------------------- #
#                       mutual information classificator                       #
# ---------------------------------------------------------------------------- #
struct MutualInformationClassif{T <: AbstractLimiter} <: AbstractMutualInformationClassif{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMutualInformationClassif) = true
is_unsupervised(::AbstractMutualInformationClassif) = false

function score(X::AbstractArray, y::Vector{Int64}, selector::MutualInformationClassif)
    return mutual_info_classif(X, y)
end

# Ranking
MutualInformationClassifRanking(nbest) = MutualInformationClassif(RankingLimiter(nbest, true))
