# ---------------------------------------------------------------------------- #
#                               fisher filter                                  #
# ---------------------------------------------------------------------------- #
struct FisherScoreFilter{T <: AbstractLimiter} <: AbstractFisherScore{T}
    limiter::T
    # parameters
end

is_supervised(::AbstractFisherScore) = true
is_unsupervised(::AbstractFisherScore) = false

function score(
    X::AbstractMatrix,
    y::AbstractVector{<:Class},
    selector::FisherScoreFilter
)::Vector{Float64}
    lmy = labelmap(y)
    ey = labelencode(lmy, y)
    scores = fisher_score.fisher_score(X, ey)
    return scores
end
score(Xdf::AbstractDataFrame, y::AbstractVector{<:Class}, selector::FisherScoreFilter) = score(Matrix(Xdf), y, selector)