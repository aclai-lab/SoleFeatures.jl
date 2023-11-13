struct FisherScoreFilter{T <: AbstractLimiter} <: AbstractFisherScore{T}
    limiter::T
    # parameters
end

# ========================================================================================
# TRAITS

is_supervised(::AbstractFisherScore) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::FisherScoreFilter
)::Vector{Float64}
    lmy = labelmap(y)
    ey = labelencode(lmy, y)
    scores = fisher_score.fisher_score(Matrix(X), ey)
    return scores
end
