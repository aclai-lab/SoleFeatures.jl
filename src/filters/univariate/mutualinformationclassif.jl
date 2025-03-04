# ---------------------------------------------------------------------------- #
#                       mutual information classificator                       #
# ---------------------------------------------------------------------------- #
struct MutualInformationClassif{T <: AbstractLimiter} <: AbstractMutualInformationClassif{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMutualInformationClassif) = true
is_unsupervised(::AbstractMutualInformationClassif) = false

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::MutualInformationClassif
)::Vector{Float64}
    scores = mutual_info_classif(Matrix(X), y)
    return scores
end

function score(
    X::AbstractMatrix,
    y::AbstractVector{<:Class},
    selector::MutualInformationClassif
)::Vector{Float64}
    scores = mutual_info_classif(Matrix(X), y)
    return scores
end

# Ranking
MutualInformationClassifRanking(nbest) = MutualInformationClassif(RankingLimiter(nbest, true))
