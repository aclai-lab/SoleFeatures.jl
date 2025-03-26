# ---------------------------------------------------------------------------- #
#                                chi2 filter                                   #
# ---------------------------------------------------------------------------- #
struct Chi2Filter{T<:AbstractLimiter} <: AbstractChi2Filter{T}
    limiter::T
    # parameters
end

is_supervised(::AbstractChi2Filter) = true
is_unsupervised(::AbstractChi2Filter) = false

# ========================================================================================
# SCORE

function score(
    X::AbstractMatrix,
    y::AbstractVector{<:Class},
    selector::Chi2Filter
)::Vector{Float64}
    numcol = size(X, 2)
    scores = Vector{AbstractFloat}(undef, numcol)
    lmy = labelmap(y)
    ey = labelencode(lmy, y)
    for i in 1:numcol
        col = X[:, i]
        lmx = labelmap(col)
        ex = labelencode(lmx, col)
        r = (1:length(lmx), 1:length(lmy))
        scores[i] = pvalue(ChisqTest(ex, ey, r))
    end
    return scores
end
score(Xdf::AbstractDataFrame, y::AbstractVector{<:Class}, selector::Chi2Filter) = score(Matrix(Xdf), y, selector)

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
Chi2Threshold(; alpha = 0.05) = Chi2Filter(ThresholdLimiter(alpha, ≤))
Chi2Ranking(nbest) =  Chi2Filter(RankingLimiter(nbest, false))
