# ---------------------------------------------------------------------------- #
#                       mutual information classificator                       #
# ---------------------------------------------------------------------------- #
struct MutualInformationClassif{T <: AbstractLimiter} <: AbstractMutualInformationClassif{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMutualInformationClassif) = true

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::MutualInformationClassif
)::Vector{Float64}
    scores = fs.mutual_info_classif(Matrix(X), y)
    return scores
end

# Ranking
MutualInformationClassifRanking(nbest) = MutualInformationClassif(RankingLimiter(nbest, true))
