struct MutualInformationClassif{T <: AbstractLimiter} <: AbstractMutualInformationClassif{T}
    limiter::T
    # parameters
end

# ========================================================================================
# TRAITS

is_supervised(::AbstractMutualInformationClassif) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::MutualInformationClassif
)::Vector{Float64}
    scores = fs.mutual_info_classif(Matrix(X), y)
    return scores
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

MutualInformationClassifRanking(nbest) =
    MutualInformationClassif(RankingLimiter(nbest, true))
