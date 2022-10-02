struct CorrelationRanking <: AbstractFilterBased
    k::Int64
    cor_algorithm::Symbol
    memorysaving::Bool

    CorrelationRanking(k::Int64, cor_algorithm::Symbol, memorysaving::Bool) =
        new(k, cor_algorithm, memorysaving)

    function CorrelationRanking(k::Int64, cor_algorithm::Symbol)
        k < 0 && throw(ErrorException("k must be greater or equal 0"))
        if !(cor_algorithm in [:pearson, :spearman, :kendall])
            throw(ErrorException("cor_algorithm must be :pearson, :spearman, :kendall"))
        end
        new(k, cor_algorithm, false)
    end

end

selector_k(selector::CorrelationRanking) = selector.k
function selector_rankfunct(selector::CorrelationRanking)
    if selector.cor_algorithm == :pearson
        return StatsBase.cor
    elseif selector.cor_algorithm == :spearman
        return StatsBase.corspearman
    elseif selector.cor_algorithm == :kendall
        return StatsBase.corkendall
    end
end
selector_memorysaving(selector::CorrelationRanking) = selector.memorysaving

function build_bitmask(df::AbstractDataFrame, selector::CorrelationRanking)::BitVector
    n_cols = ncol(df)
    k = selector_k(selector)
    rf = selector_rankfunct(selector)
    ms = selector_memorysaving(selector)
    mtrx = Matrix(df)
    dims = [ maximum(ndims.(c)) for c in eachcol(mtrx) ]
    d = dims[1]

    !all(==(d), dims) && throw(DimensionMismatch("different dimensions not allowed"))
    k >= n_cols && return trues(n_cols)

    if (d == 0)
        cormtrx = rf(mtrx)
    elseif (d == 1)
        cormtrx = ms ? _correlation_dtw_memory_saving(mtrx, rf) : rf(_compute_dtw(df))
    else
        throw("Unimplemented for dimension >1")
    end

    idxes = findcorrelation(cormtrx)

    bm = falses(n_cols)
    bm[idxes[1:k]] .= true
    return bm
end
