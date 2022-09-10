"""
    _compute_dtw(df)

Compute DTW between each timeseries in a column for each column of `df`

Returns a matrix of nr*(nr-1)/2 rows (with nr number of rows in `df`) and nc columns (with nc number of columns in `df`)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame on which to calculate DTW
"""
function _compute_dtw(df::AbstractDataFrame)::Array{Float64,2}
    nr, nc = size(df)
    # number of rows in the result matrix
    nrm = Int((nr * (nr - 1)) / 2)
    # distances matrix
    dtwmtrx = Array{Float64,2}(undef, nrm, nc)

    # computation of the dtw for each timeseries in a column for each attribute in df
    Threads.@threads for cidx in 1:nc
        dtwvector = _compute_dtw(df[!, cidx])
        dtwmtrx[!, cidx] = dtwvector
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
            blk = len - i
            nump = Int(blk * (blk + 1) / 2)
            startidx = newlen - nump
            idx = startidx + (j - i)
            # dtw returns cost and a set of indices (i1,i2) that align the two serie, so only cost (dtw(...)[1])
            # have to be extracted
            cost = dtw(v[i], v[j])[1]
            distances[idx] = cost
        end
    end
    return distances
end

function _correlation_memory_saving(df::AbstractDataFrame, corf::Function)
    nc = ncol(df)
    cormtrx = Matrix{Float64}(I, nc, nc)
    for i in 1:(nc-1)
        col = _compute_dtw(df[:, i])
        for j in (i+1):nc
            compcol = _compute_dtw(df[:, j])
            cormtrx[i,j] = cormtrx[j,i] = corf(col, compcol)
        end
    end
    return cormtrx
end

"""
    correlation(df, corf)

Returns mean absolute correlation vector, based on dtw

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame on which to calculate mean absolute correlation vector
- `corf::Function`: correlation function, function that generates the correlation matrix
- `memorysaving::Bool`: calculate dtw each time for each column, should only be used in a large dataset (does not affect `df` with zero dimension data)
"""
function correlation(
    df::AbstractDataFrame,
    corf::Function;
    memorysaving::Bool=false
)::Array{Float64}
    dim = SoleBase.dimension(df)
    if (dim == 0)
        cormtrx = corf(Matrix(df))
    elseif (dim == 1)
        cormtrx = memorysaving ?
            _correlation_memory_saving(df, corf) : corf(_compute_dtw(df))
    else
        throw("Unimplemented for dimension >1")
    end

    # absolute value of each correlation coefficient
    cormtrx = abs.(cormtrx)
    # NaN values obtained from equal time series are converted into 1 (max correlation)
    replace!(cormtrx, NaN => 1)
    # calculate avg of correlation matrix per column (mean absolute correlation vector)
    avg_vector = vec(mean(cormtrx, dims=1))
    return avg_vector
end
