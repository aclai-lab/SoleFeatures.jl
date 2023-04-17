struct Chi2Filter{T <: AbstractLimiter} <: AbstractChi2Filter{T}
    limiter::T
    # parameters
end

# ========================================================================================
# TRAITS

is_supervised(::AbstractChi2Filter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
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

# ========================================================================================
# CUSTOM CONSTRUCTORS

ThresholdChi2(; alpha = 0.05) = Chi2Filter(ThresholdLimiter(alpha, <=))
RankingChi2(nbest) =  Chi2Filter(RankingLimiter(nbest, false))
