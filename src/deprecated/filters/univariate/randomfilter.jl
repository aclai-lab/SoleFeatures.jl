struct RandomFilter{T<:AbstractLimiter} <: AbstractRandomFilter{T}
    limiter::T
    # parameters
    seed::Union{Int,Nothing}
end

# ========================================================================================
# ACCESSORS

seed(selector::RandomFilter) = selector.seed

# ========================================================================================
# TRAITS

is_unsupervised(::AbstractRandomFilter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    selector::RandomFilter
)::Vector{<:Real}
    s = seed(selector)
    rng = isnothing(s) ? MersenneTwister() : MersenneTwister(s)
    return rand(rng, ncol(X))
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

RandomRanking(nbest::Integer, seed::Integer) = RandomFilter(RankingLimiter(nbest), seed)
RandomRanking(nbest::Integer) = RandomFilter(RankingLimiter(nbest), nothing)
