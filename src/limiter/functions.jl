threshold(tl::ThresholdLimiter) = tl.threshold
ford(tl::ThresholdLimiter) = tl.ordf

function apply_limiter(scores::AbstractVector{<:Real}, tl::ThresholdLimiter)
    return findall(ford(tl)(threshold(tl)), scores)
end

nbest(rl::RankingLimiter) = rl.nbest
rev(rl::RankingLimiter) = rl.rev

function apply_limiter(scores::AbstractVector{<:Real}, rl::RankingLimiter)
    return sortperm(scores; rev=rev(rl))[1:nbest(rl)]
end

suiteness(gfl::GroupFittestLimiter) = gfl.suiteness

function apply_limiter(
    winboard::AbstractVector{BitVector},
    gfl::GroupFittestLimiter
)
    return findall([sum(bm) >= ceil(Int, length(bm) * suiteness(gfl)) for bm in winboard])
end

function apply_limiter(
    winboard::AbstractVector{BitVector},
    goil::GroupOneInLimiter
)
    wins = sum(winboard; dims=1)[1]
    return findall(>=(1), wins)
end
