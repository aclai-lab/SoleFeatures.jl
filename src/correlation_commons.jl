"""
    _compute_dtw(df)

Compute DTW between each timeseries in a column for each column of `df`

Returns a matrix of nr*(nr-1)/2 rows (with nr number of rows in `df`) and nc columns (with nc number of columns in `df`)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame on which to calculate DTW
"""
# _compute_dtw(mtrx::AbstractMatrix; dim=1) = __compute_dtw(mtrx, dim)
_compute_dtw(df::AbstractDataFrame) = _compute_dtw(Matrix(df))

function _compute_dtw(mtrx::AbstractMatrix)::Array{Float64,2}
    dim = 1 # transofrm in parameter
    ns = collect(size(mtrx))
    ns[dim] = Int((ns[dim] * (ns[dim] - 1)) / 2)
    # distances matrix
    dtwmtrx = Array{Float64,2}(undef, ns...)

    # computation of the dtw for each timeseries in a column for each attribute in df
    d = size(mtrx, 2)
    Threads.@threads for idx in 1:d
        index = (:, idx)
        dtwvector = _compute_dtw(mtrx[index...])
        dtwmtrx[index...] = dtwvector
    end

    return dtwmtrx
end

function _compute_dtw(v::AbstractVector)::Vector{Float64}
    len = length(v)
    newlen = Int((len * (len - 1)) / 2)
    distances = zeros(Float64, newlen) # dtw vector
    Threads.@threads for i in 1:(len-1)
        Threads.@threads for j in (i+1):len
            # calculate index in distances vector
            # blk = len - i
            # nump = Int(blk * (blk + 1) / 2)
            # startidx = newlen - nump
            startidx = Int((2*len - i) * (i - 1) / 2)
            idx = startidx + (j - i)
            # dtw returns cost and a set of indices (i1,i2) that align the two serie, so only cost (dtw(...)[1])
            # have to be extracted
            cost = dtw(v[i], v[j])[1]
            distances[idx] = cost
        end
    end
    return distances
end

function _correlation_dtw_memory_saving(mtrx::AbstractMatrix, corf::Function)
    nc = size(mtrx, 2)
    cormtrx = Matrix{Float64}(I, nc, nc)
    for i in 1:(nc-1)
        col = _compute_dtw(mtrx[:, i])
        for j in (i+1):nc
            compcol = _compute_dtw(mtrx[:, j])
            cormtrx[i,j] = cormtrx[j,i] = corf(col, compcol)
        end
    end
    return cormtrx
end

function _old_findcorrelation(
    cormtrx::AbstractMatrix;
    nbest::Union{Nothing, Integer}=nothing,
    exact::Bool=false,
    returncorvect::Bool=false
)
    nr, nc = size(cormtrx)

    (nr != nc) && throw(DimensionMismatch("provided not valid correlation matrix"))
    !isnothing(nbest) && (0 <= nbest > nr) &&
        throw("nbest must be <= of cormatrx dim and > 0")

    isnothing(nbest) && (nbest = nr)

    # absolute value of each correlation coefficient
    cormtrx = abs.(cormtrx)

    if (exact)
        macv = []
        macvidx = []
        oidxes = collect(1:nr)
        for i in 1:nbest
            m = vec(mean(view(cormtrx, Not(macvidx), Not(macvidx)), dims=1))
            cbidx = sortperm(m)[1]
            bidx = view(oidxes, Not(macvidx))[cbidx]
            push!(macv, m[cbidx])
            push!(macvidx, bidx)
        end
    else
        macv = vec(mean(cormtrx, dims=1))
        macvidx = sortperm(macv)[1:nbest]
    end

    return ( returncorvect ? (macvidx, macv) : macvidx )
end

"""
    findcorrelation(cormtrx; threshold)
"""
function findcorrelation(cormtrx::AbstractMatrix; threshold::AbstractFloat=0.0)
    nr, nc = size(cormtrx)

    (nr != nc) && throw(DimensionMismatch("provided not valid correlation matrix"))
    (0.0 < threshold > 1.0) && throw(ArgumentError("threshold must be between 0.0 and 1.0"))

    cormtrx = abs.(cormtrx)
    macv = vec(mean(cormtrx, dims=1)) # mean absolute correlation vector
    # preparing correlation matrix
    cormtrx[diagind(cormtrx)] .= -Inf
    cormtrx = UpperTriangular(cormtrx)
    oidxes = collect(1:nr) # original indices
    cvidx = [] # correlation vector indices

    for _ in 1:nr
        vcormtrx = view(cormtrx, Not(cvidx), Not(cvidx))
        mc1, mc2 = Tuple(findmax(vcormtrx)[2]) # indices of most correlated attributes
        vcormtrx[mc1, mc2] < threshold && break
        mcidx = macv[mc1] >= macv[mc2] ? mc1 : mc2
        bidx = view(oidxes, Not(cvidx))[mcidx]
        push!(cvidx, bidx)
    end

    return iszero(threshold) ? reverse(cvidx) : setdiff(oidxes, cvidx)
end
