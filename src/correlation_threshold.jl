struct CorrelationThreshold <: AbstractFilterBased
    threshold::Float64
    cor_algorithm::Symbol

    function CorrelationThreshold(threshold::Float64, cor_algorithm::Symbol)
        if threshold < 0.0
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        end
        if !(cor_algorithm in [:pearson :spearman :kendall])
            throw(ErrorException("cor_algorithm must be :pearson, :spearman, :kendall"))
        end
        new(threshold, cor_algorithm)
    end
end

threshold(selector::CorrelationThreshold) = selector.threshold
function corfunction(selector::CorrelationThreshold)
    if selector.cor_algorithm == :pearson
        return StatsBase.cor
    elseif selector.cor_algorithm == :spearman
        return StatsBase.corspearman
    elseif selector.cor_algorithm == :kendall
        return StatsBase.corkendall
    end
end

function build_bitmask(df::AbstractDataFrame, selector::CorrelationThreshold)::BitVector
    n_cols = ncol(df)
    thr = threshold(selector)
    cf = corfunction(selector)
    ms = selector_memorysaving(selector)
    mtrx = Matrix(df)
    dims = [ maximum(ndims.(c)) for c in eachcol(mtrx) ]
    d = dims[1]

    !all(==(d), dims) && throw(DimensionMismatch("different dimensions not allowed"))

    # return immediately if 'k' is greater than columns number
    k > n_cols && return trues(ncol)

    if (d == 0)
        cormtrx = cf(mtrx)
    elseif (d == 1)
        cormtrx = ms ? _correlation_memory_saving(mtrx, cf) : cf(_compute_dtw(df))
    else
        throw("Unimplemented for dimension >1")
    end

    idxes = findcorrelation(cormtrx, threshold=thr)

    bm = falses(n_cols)
    bm[idxes] .= true
    return bm
end
