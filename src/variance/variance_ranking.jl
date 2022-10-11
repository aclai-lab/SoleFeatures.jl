struct VarianceRanking <: AbstractFilterBased
    nbest::Integer

    function VarianceRanking(nbest::Integer)
        nbest < 0 && throw(DomainError(nbest, "'nbest' must be greater or equal 0"))
        new(nbest)
    end
end

# traits
is_multivariate(::VarianceRanking) = true
is_unsupervised(::VarianceRanking) = true

# getter
nbest(selector::VarianceRanking) = selector.nbest

function apply(df::AbstractDataFrame, selector::VarianceRanking)::Vector{Integer}
    n_cols = ncol(df)
    k = nbest(selector)

    k >= n_cols && return trues(n_cols)

    vars = [ StatsBase.var(Iterators.flatten(c)) for c in eachcol(df) ]
    replace!(vars, NaN => -Inf)
    bestidxes = sortperm(vars; rev=true)[1:k]
    return bestidxes
end
