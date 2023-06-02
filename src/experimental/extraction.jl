struct DescribedDataFrame{T} <: AbstractDataFrame
    df::DataFrame
    descriptors::Dict{T, Int}
end

const _SEPARATOR = "@@@"

"""
First element must be the name of the variable
"""
const Extractor = Tuple{Union{String, Symbol}, Vararg{Any}}

@inline iscallable(::Function) = true
@inline iscallable(c::Any) = !isempty(methods(c))

function _extract(v::AbstractVector, e::Extractor)
    res = deepcopy(v)
    for (i, item) in enumerate(e)
        (isa(item, Symbol) || isa(item, String)) && continue
        !iscallable(item) && throw(ErrorException("Extractor contains not callable object at index $(i): $(item)"))
        res = item.(res)
    end
    return res
end

function extract(df::AbstractDataFrame, es::Array{<:Extractor})
    return DataFrame(string.(es) .=> _extract.(getindex.([df], :, getindex.(es, 1)), es))
end

function groupby(es::Array{<:Extractor}, idxes::Union{Int, NTuple{N, Int}}) where {N}
    res = Dict{Extractor, Vector{Extractor}}()
    for e in es
        push!(get!(res, keepat(e, idxes), Vector{Extractor}()), e)
    end
    return [ values(res)... ]
end

function representatives(es::Array{<:Extractor}, idxes::Union{Int, NTuple{N, Int}}) where {N}
    return unique(keepat.(es, [ idxes ]))
end

function keepat(e::Extractor, idxes::Union{Int, NTuple{N, Int}}) where {N}
    return getindex(e, [idxes...])
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
