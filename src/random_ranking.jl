struct RandomRanking <: AbstractFilterBased
    nbest::Integer
    rnd_seed::Union{Integer,Nothing}

    RandomRanking(nbest::Integer) = new(nbest, nothing)
    RandomRanking(nbest::Integer, rnd_seed::Integer) = new(nbest, rnd_seed)
end

# traits
is_univariate(::RandomRanking) = true
is_unsupervised(::RandonRanking) = true

# getter
nbest(selector::RandomRanking) = selector.nbest
seed(selector::RandomRanking) = selector.rnd_seed

function apply(df::AbstractDataFrame, selector::RandomRanking)::Vector{Integer}
    s = seed(selector)
    rng = isnothing(s) ? MersenneTwister() : MersenneTwister(s)
    idxes = StatsBase.sample(rng, 1:ncol(df), nbest(selector); replace=false)
    return idxes
end
