struct RandomCriterion <: AbstractScalarCriterion
    seed::Union{Int, Nothing}
end

# ========================================================================================
# ACCESSORS

seed(c::RandomCriterion) = c.seed

# ========================================================================================
# TRAITS

issupervised(::RandomCriterion) = false
isunivariate(::RandomCriterion) = false

# ========================================================================================
# SCORES

function scores(c::RandomCriterion, X::AbstractDataFrame)::Vector{<:Real}
    s = seed(c)
    rng = isnothing(s) ? MersenneTwister() : MersenneTwister(s)
    return rand(rng, ncol(X))
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

RandomRanking(nbest::Integer, seed::Integer) = RandomFilter(RankingLimiter(nbest), seed)
RandomRanking(nbest::Integer) = RandomFilter(RankingLimiter(nbest), nothing)
