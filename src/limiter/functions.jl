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

suiteness(fl::FittestLimiter) = fl.suiteness

function apply_limiter(
    winboard::AbstractVector{AbstractVector{<:Bool}},
    fl::FittestLimiter
)
    nwin = ceil(length(winboard) * suiteness(fl))
    wins = sum(winboard; dims=1)[1]
    return findall(>=(nwin), wins)
end
