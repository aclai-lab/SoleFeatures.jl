import StatsBase: transform!, transform

"""
    transform!(X, args..; kwargs...)
    transform!(X, y, args..; kwargs...)

Remove from provided samples variables indicated by bitmask or selector

# Arguments

- `X::AbstractDataFrame|AbstractMultiModalDataset`: samples to evaluate
- `y::AbstractVector`: target vector
- `bm::BitVector`: Vector of bit containing which variables are suitable(1) or not(0)
- `selector::AbstractFeatureSelector`: Selector (TODO fix)

# Keywords

- `i_modality::Integer`: Index of the modality in `X` (TODO fix)
"""
function transform!(X::AnyDataset, idxes::Vector{Int})
    size(X, 2) < length(idxes) && throw(DimensionMismatch(""))
    return DataFrame.select!(X, idxes)
end

function transform!(X::AnyDataset, bm::BitVector)
    size(X, 2) != length(bm) && throw(DimensionMismatch(""))
    return DataFrame.select!(X, bm)
end

function transform!(selector::AbstractFeatureSelector, X::AnyDataset)
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
    i_modality::Union{Integer, Nothing} = nothing
)
    if (isnothing(i_modality))
        nvariables(X) != length(bm) && throw(DimensionMismatch(""))
        return SoleData.dropvariables!(X, findall(!, bm))
    else
        nvariables(X, i_modality) != length(bm) && throw(DimensionMismatch(""))
        return SoleData.dropvariables!(X, i_modality, findall(!, bm))
    end
end

function transform!(
    selector::AbstractFeatureSelector,
    X::SoleData.AbstractMultiModalDataset;
    i_modality::Union{Integer, Nothing} = nothing
)
    if (isnothing(i_modality))
        return transform!(SoleData.data(X), selector)
    else
        return transform!(SoleData.modality(X, i_modality), selector)
    end
end

# TODO: transform! for AbstractMultiModalDataset with supervised selector

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
    buildbitmask(selector, X, i_modality)

return a bitmask containing selected variables from selector.
True values indicate selected variable index

# Arguments

- `selector::AbstractFeatureSelector`: applied selector
- `X::AbstractDataFrame`: samples to evaluate
- `y::AbstractVector{<:Class}`: target vector

# Keywords

- `i_modality::Integer`: Index of the modality in `X`
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

# TODO: buildbitmask for AbstractMultiModalDataset with supervised selector
function buildbitmask(
    selector::AbstractFeatureSelector,
    X::SoleData.AbstractMultiModalDataset,
    i_modality::Integer
)::Tuple{BitVector, BitVector}
    return buildbitmask(SoleData.modality(X, i_modality), selector)
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
