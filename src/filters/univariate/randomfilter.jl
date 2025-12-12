# ---------------------------------------------------------------------------- #
#                               random filter                                  #
# ---------------------------------------------------------------------------- #
"""
    RandomFilter{T <: AbstractLimiter} <: AbstractRandomFilter{T}

A unsupervised univariate feature selection filter that uses Mersenne Twister 
algorithm to generate random scores, one for each feature.

# Fields
- `limiter::T`: A limiter that defines the selection criterion to be applied to scores.
- `seed::Union{Int,Nothing}`: An optional integer seed for the random number generator.
"""
struct RandomFilter{T<:AbstractLimiter} <: AbstractRandomFilter{T}
    limiter::T
    # parameters
    seed::Union{Int,Nothing}
end

seed(selector::RandomFilter) = selector.seed

is_supervised(::AbstractRandomFilter) = false
is_unsupervised(::AbstractRandomFilter) = true

"""
    score(selector, X)

Compute random scores for each feature in `X`.

# Arguments
- `selector::RandomFilter`: Instance of RandomFilter
- `X::AbstractArray`: Feature matrix (n_samples × n_features)

# Algorithm
1. Check that all columns in `X` are of type `Real`
2. For each feature, compute Pearson correlation coefficient with target `y`

# Example
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
scores = SoleFeatures.score(RandomFilter(RankingLimiter(3, false), nothing), X)
```
"""
function score(selector::RandomFilter, X::AbstractArray)::Vector{Float64}
    s = seed(selector)
    rng = isnothing(s) ? MersenneTwister() : MersenneTwister(s)
    return rand(rng, size(X, 2))
end

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
get_random_identity(seed::Union{Int,Nothing}=nothing) = RandomFilter(IdentityLimiter(), seed)
get_random_threshold(threshold::Real, ordf::Function, seed::Union{Int,Nothing}=nothing) = RandomFilter(ThresholdLimiter(threshold, ordf), seed)
get_random_ranking(nbest::Integer, rev::Bool=false, seed::Union{Int,Nothing}=nothing) = RandomFilter(RankingLimiter(nbest, rev), seed)
get_random_percentage(percentage::Real, rev::Bool=true, seed::Union{Int,Nothing}=nothing) = RandomFilter(PercentageLimiter(percentage, rev), seed)
