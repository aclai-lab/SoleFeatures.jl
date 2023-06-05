struct CorrelationFilter <: AbstractCorrelationFilter
    corf::Function
    threshold::Real

    function CorrelationFilter(corf::Function, threshold::Real)
        if (threshold < 0 || threshold > 1)
            throw(DomainError("Threshold must be within 0 and 1"))
        end
        return new(corf, threshold)
    end
end

# ========================================================================================
# ACCESSORS

corf(selector::CorrelationFilter) = selector.corf
threshold(selector::CorrelationFilter) = selector.threshold

# ========================================================================================
# TRAITS

is_unsupervised(::AbstractCorrelationFilter) = true

# ========================================================================================
# APPLY

function apply(
    X::AbstractDataFrame,
    selector::CorrelationFilter
)::Vector{Int}
    mtrx = Matrix(X)
    cormtrx = corf(selector)(mtrx)
    return findcorrelation(cormtrx; threshold=threshold(selector))
end

# ========================================================================================
# UTILS

"""
    findcorrelation(cormtrx; threshold)
"""
function findcorrelation(cormtrx::AbstractMatrix; threshold::Real = 0)
    nr, nc = size(cormtrx)

    (nr != nc) && throw(DimensionMismatch("provided not valid correlation matrix"))
    (0.0 < threshold > 1.0) && throw(ArgumentError("threshold must be between 0.0 and 1.0"))

    cormtrx = abs.(cormtrx)
    macv = vec(mean(cormtrx, dims=1)) # mean absolute correlation vector
    # preparing correlation matrix
    cormtrx[diagind(cormtrx)] .= 0
    cormtrx = UpperTriangular(cormtrx)
    oidxes = collect(1:nr) # original indices
    cvidx = [] # correlation vector indices

    for _ in 1:nr
        vcormtrx = view(cormtrx, Not(cvidx), Not(cvidx))
        mc1, mc2 = Tuple(findmax(vcormtrx)[2]) # indices of most correlated variables
        vcormtrx[mc1, mc2] < threshold && break
        mcidx = macv[mc1] >= macv[mc2] ? mc1 : mc2
        bidx = view(oidxes, Not(cvidx))[mcidx]
        push!(cvidx, bidx)
    end

    return iszero(threshold) ? reverse(cvidx) : setdiff(oidxes, cvidx)
end
