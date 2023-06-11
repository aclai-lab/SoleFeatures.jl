"""
    transform!(X, args..; kwargs...)
    transform!(X, y, args..; kwargs...)

Removes from provided samples variables indicated by bitmask or selector

# Arguments

- `X::AbstractDataFrame|MultiModalDataset`: samples to evaluate
- `y::AbstractVector`: target vector
- `bm::BitVector`: Vector of bit containing which variables are suitable(1) or not(0)
- `selector::AbstractFeatureSelector`: Selector

# Keywords

- `mdlidx::Integer`: Frame index inside `X`
"""
function transform!(X::Dataset, idxes::Vector{Int})
    size(X, 2) < length(idxes) && throw(DimensionMismatch(""))
    return DataFrame.select!(X, idxes)
end

function transform!(X::Dataset, bm::BitVector)
    size(X, 2) != length(bm) && throw(DimensionMismatch(""))
    return DataFrame.select!(X, bm)
end

function transform!(selector::AbstractFeatureSelector, X::Dataset)
    return transform!(X, selector(X))
end

function transform!(
    selector::AbstractFeatureSelector,
    X::AbstractDataFrame,
    y::AbstractVector{<:Class}
)
    return transform!(X, selector(X, y))
end

function transform!(
    X::SoleData.AbstractMultiModalDataset,
    bm::BitVector;
    mdlidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(mdlidx))
        nvariables(X) != length(bm) && throw(DimensionMismatch(""))
        return SoleData.dropvariables!(X, findall(!, bm))
    else
        nvariables(X, mdlidx) != length(bm) && throw(DimensionMismatch(""))
        return SoleData.dropvariables!(X, mdlidx, findall(!, bm))
    end
end

function transform!(
    selector::AbstractFeatureSelector,
    X::SoleData.AbstractMultiModalDataset;
    mdlidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(mdlidx))
        return transform!(SoleData.data(X), selector)
    else
        return transform!(SoleData.modality(X, mdlidx), selector)
    end
end

# TODO: transform! for MultiModalDataset with supervised selector

transform(
    selector::AbstractFeatureSelector,
    X::AbstractDataFrame,
    args...;
    kwargs...
) = transform!(selector, deepcopy(X), args...; kwargs...)

transform(
    selector::AbstractFeatureSelector,
    X::SoleData.AbstractMultiModalDataset,
    args...;
    kwargs...
) = transform!(selector, deepcopy(X), args...; kwargs...)

"""
    buildbitmask(selector, X)
    buildbitmask(selector, X, y)
    buildbitmask(selector, X, mdlidx)

return a bitmask containing selected variables from selector.
True values indicate selected variable index

# Arguments

- `selector::AbstractFeatureSelector`: applied selector
- `X::AbstractDataFrame`: samples to evaluate
- `y::AbstractVector{<:Class}`: target vector

# Keywords

- `mdlidx::Integer`: Frame index inside `X`
"""
function buildbitmask(
    selector::AbstractFeatureSelector,
    X::AbstractDataFrame
)::BitVector
    return _idxes2bm(size(X, 2), selector(X))
end

function buildbitmask(
    selector::AbstractFeatureSelector,
    X::AbstractDataFrame,
    y::AbstractVector{<:Class}
)::BitVector
    return _idxes2bm(size(X, 2), selector(X, y))
end

# TODO: buildbitmask for MultiModalDataset with supervised selector
function buildbitmask(
    selector::AbstractFeatureSelector,
    X::SoleData.MultiModalDataset,
    mdlidx::Integer
)::Tuple{BitVector, BitVector}
    return buildbitmask(SoleData.modality(X, mdlidx), selector)
end

"""
    _idxes2bm(bmlen, idxes)

return bit vector containing trues in indicated indices

# Example

```jldoctest
julia> indices = [8,5,2]
3-element Vector{Int64}:
 8
 5
 2

julia> _idxes2bm(10, indices)
10-element BitVector:
 0
 1
 0
 0
 1
 0
 0
 1
 0
 0
 ```

"""
function _idxes2bm(bmlen::Integer, idxes::AbstractVector{<:Integer})
    bm = falses(bmlen)
    bm[idxes] .= true
    return bm
end
