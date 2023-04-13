struct DescribedDataFrame{T} <: AbstractDataFrame
    df::DataFrame
    descriptors::Dict{T, Int}
end

const _SEPARATOR = "@@@"

# TODO: remove Any in Vararg, Any is defined only for test purpose using WindowsIndex.
#       Understand why a callable object is not a function
const Extractor = Tuple{Union{String, Symbol}, Vararg{Union{Function, Symbol, String, Any}}}

function extract(v::AbstractVector, e::Extractor)
    res = []
    for item in e
        (isa(item, Symbol) || isa(item, String)) && continue
        res = item.(v)
    end
    return res
end

function extract(df::AbstractDataFrame, es::Array)
    edf = DataFrame()
    for e in es
        colname = string(e)
        colvals = extract(df[:, e[1]], e)
        edf[!, colname] = colvals
    end
    return edf
end

function groupby(es::Array, group::Union{Int, NTuple{N, Int}}) where {N}
    groups = Dict()
    for e in es
        length(group) > length(e) && throw(DimensionMismatch("`group` have more elements than current `ds` item: $(e)"))
        sel = e[[group...]]
        if (haskey(groups, sel))
            push!(groups[sel], e)
        else
            groups[sel] = Vector{Extractor}([ e ])
        end
    end
    return [ values(groups)... ]
end

function groupreduce(es::Array, group::Union{Int, NTuple{N, Int}}) where {N}
    groups = groupby(es, group)
    res = []
    for t in getindex.(groups, 1)
        push!(res, t[[group...]])
    end
    return res
end

function Base.string(d::Extractor)
    str = ""
    for (i, x) in enumerate(d)
        if (i != 1) str *= _SEPARATOR end
        if (isa(x, Function)) x = nameof(x) end
        str *= string(x)
    end
    return str
end
