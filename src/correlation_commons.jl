# DTW AVG Correlation
# consider using : https://docs.julialang.org/en/v1/stdlib/Mmap/ for large dataset

"""
    _compute_dtw(df)

Compute DTW between each timeseries in a column for each column of `df`

Returns a matrix of nr*(nr-1)/2 rows (with nr number of rows in `df`) and nc columns (with nc number of columns in `df`)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame on which to calculate DTW
"""
function _compute_dtw(df::AbstractDataFrame)::Array{Float64,2}
    # maybe a better implmentation do dtw only on a vector of timeseries

    nr, nc = size(df)
    # number of rows in the result matrix
    nrm = Int((nr * (nr - 1)) / 2)
    # distances matrix
    dist_matrix = Array{Float64,2}(undef, nrm, nc)
    # computation of the dtw for each timeseries in a column for each attribute in df
    Threads.@threads for cidx in 1:nc
        idxm = 1
        for iidx in 1:(nr-1)
            for jidx in (iidx+1):nr
                # dtw returns cost and a set of indices (i1,i2) that align the two serie, so only cost (dtw(...)[1])
                # have to be extracted
                dist_matrix[idxm, cidx] = dtw(df[iidx, cidx], df[jidx, cidx])[1]
                idxm = idxm + 1
            end
        end
    end
    return dist_matrix
end

function _compute_dtw(v::AbstractVector)::Vector{Float64}
    distances = Vector{Float64}() # dtw vector
    len = length(v)
    for i in 1:(len-1)
        for j in (i+1):len
            # dtw returns cost and a set of indices (i1,i2) that align the two serie, so only cost (dtw(...)[1])
            # have to be extracted
            cost = dtw(v[i], v[j])[1]
            push!(distances, cost)
        end
    end
    return distances
end

"""
    correlation(df, corf)

Returns mean absolute correlation vector, based on dtw

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame on which to calculate mean absolute correlation vector
- `corf::Function`: correlation function, function that generates the correlation matrix
- `memory_saving::Bool`: calculate dtw each time for each column, should only be used in a large dataset (does not affect `df` with zero dimension data)
"""
function correlation(
    df::AbstractDataFrame,
    corf::Function;
    memory_saving::Bool=false
)::Array{Float64}
    df_dim = SoleBase.dimension(df)
    nc = ncol(df)
    cor_matrix = Matrix{Float64}(I, nc, nc)

    if (df_dim == 0)
        # correlation matrix built from the correlation function provided (corf)
        # absolute value of each coefficient is calculated
        cor_matrix = corf(Matrix(df))
    elseif (df_dim == 1)
        if (memory_saving)
            for i in 1:(nc-1)
                col = _compute_dtw(df[:, i])
                for j in (i+1):nc
                    compcol = _compute_dtw(df[:, j])
                    cor_matrix[i,j] = cor_matrix[j,i] = corf(col, compcol)
                end
            end
        else
            cor_matrix = corf(_compute_dtw(df))
        end
    else
        error("unimplemented for dimension >1")
    end

    cor_matrix = abs.(cor_matrix)
    # NaN values obtained from equal time series are converted into 1
    replace!(cor_matrix, NaN => 1)
    # calculate avg of correlation matrix per column (mean absolute correlation vector)
    avg_vector = vec(mean(cor_matrix, dims=1))
    return avg_vector
end
