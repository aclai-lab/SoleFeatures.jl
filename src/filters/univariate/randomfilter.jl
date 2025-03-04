# ---------------------------------------------------------------------------- #
#                               random filter                                  #
# ---------------------------------------------------------------------------- #
struct RandomFilter{T<:AbstractLimiter} <: AbstractRandomFilter{T}
    limiter::T
    # parameters
    seed::Union{Int,Nothing}
end

seed(selector::RandomFilter) = selector.seed

is_supervised(::AbstractRandomFilter) = false
is_unsupervised(::AbstractRandomFilter) = true

function score(X::AbstractMatrix, selector::RandomFilter)::Vector{Float64}
    s = seed(selector)
    rng = isnothing(s) ? MersenneTwister() : MersenneTwister(s)
    return rand(rng, size(X, 2))
end
score(Xdf::AbstractDataFrame, selector::RandomFilter)::Vector{Float64} = score(Matrix(Xdf), selector)

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
RandomRanking(nbest::Integer, seed::Integer) = RandomFilter(RankingLimiter(nbest), seed)
RandomRanking(nbest::Integer) = RandomFilter(RankingLimiter(nbest), nothing)
