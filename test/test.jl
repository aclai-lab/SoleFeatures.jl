using SoleFeatures
using DynamicAxisWarping

include("./test_function.jl")

function _compute_dtw(df::AbstractDataFrame; use_thread=false)::Array{Float64,2}
    # maybe a better implmentation do dtw only on a vector of timeseries

    nr, nc = size(df)
    # number of rows in the result matrix
    nrm = Int((nr * (nr - 1)) / 2)
    # distances matrix
    dist_matrix = Array{Float64,2}(undef, nrm, nc)
    if (use_thread)
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
    else
        # computation of the dtw for each timeseries in a column for each attribute in df
        for cidx in 1:nc
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
    end
    return dist_matrix
end

res = BitVector()

for i in 1:100
    df1 = random_timeseries_df()
    df2 = deepcopy(df1)
    dtw_t = _compute_dtw(df1; use_thread=true)
    dtw_nt = _compute_dtw(df2; use_thread=false)
    ri = isequal(dtw_t, dtw_nt)
    println("Finito: " * string(ri))
    push!(res, ri)
end

println(res)
println(all(==(true), res))
