struct RandomFilter{T <: AbstractFilterLimiter} <: AbstractRandomFilter{T}
    limiter::T
    # parameters
    seed::Union{Int, Nothing}
end

limiter(selector::RandomFilter) = selector.limiter
seed(selector::RandomFilter) = selector.seed

is_univariate(::AbstractRandomFilter{AbstractFilterLimiter}) = true
is_unsupervised(::AbstractRandomFilter{AbstractFilterLimiter}) = true

RandomRanking(nbest::Integer, seed::Integer) = RandomFilter(RankingLimiter(nbest), seed)
RandomRanking(nbest::Integer) = RandomFilter(RankingLimiter(nbest), nothing)

function apply(
    df::AbstractDataFrame,
    selector::RandomFilter{RankingLimiter}
)::Vector{Integer}
    s = seed(selector)
    rng = isnothing(s) ? MersenneTwister() : MersenneTwister(s)
    idxes = StatsBase.sample(rng, 1:ncol(df), nbest(limiter(selector)); replace=false)
    return idxes
end
