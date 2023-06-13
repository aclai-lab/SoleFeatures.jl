"""
    transform!(X, args..; kwargs...)
    transform!(X, y, args..; kwargs...)

Removes from provided samples attributes indicated by bitmask or selector

# Arguments

- `X::AbstractDataFrame|MultiModalDataset`: samples to evaluate
- `y::AbstractVector`: target vector
- `bm::BitVector`: Vector of bit containing which attributes are suitable(1) or not(0)
- `selector::AbstractFeaturesSelector`: Selector

# Keywords

- `frmidx::Integer`: Frame index inside `X`
"""
function transform!(X::AbstractDataFrame, idxes::Vector{Int})
    ncol(X) < length(idxes) && throw(DimensionMismatch(""))
    return select!(X, idxes)
end

function transform!(X::AbstractDataFrame, bm::BitVector)
    ncol(X) != length(bm) && throw(DimensionMismatch(""))
    return select!(X, bm)
end

function transform!(X::AbstractDataFrame, selector::AbstractFeaturesSelector)
    return transform!(X, apply(X, selector))
end

function transform!(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::AbstractFeaturesSelector
)
    return transform!(X, apply(X, y, selector))
end

function transform!(
    X::SoleData.AbstractMultiModalDataset,
    bm::BitVector;
    frmidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(frmidx))
        nvariables(X) != length(bm) && throw(DimensionMismatch(""))
        return SoleData.dropvariables!(X, findall(!, bm))
    else
        nvariables(X, frmidx) != length(bm) && throw(DimensionMismatch(""))
        return SoleData.dropvariables!(X, frmidx, findall(!, bm))
    end
end

function transform!(
    X::SoleData.AbstractMultiModalDataset,
    selector::AbstractFeaturesSelector;
    frmidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(frmidx))
        return transform!(SoleData.data(X), selector)
    else
        return transform!(SoleData.frame(X, frmidx), selector)
    end
end

# TODO: transform! for MultiModalDataset with supervised selector

transform(X::AbstractDataFrame, args...; kwargs...) = transform!(deepcopy(X), args...; kwargs...)
transform(X::SoleData.AbstractMultiModalDataset, args...; kwargs...) = transform!(deepcopy(X), args...; kwargs...)
# (s::AbstractFeaturesSelector)(X, args; kwargs...) = transform(X, args..., kwargs...) # TODO: correct this

"""
    buildbitmask(X, selector)
    buildbitmask(X, y, selector)
    buildbitmask(X, selector, frmidx)

return a bitmask containing selected attributes from selector.
True values indicate selected attribute index

# Arguments

- `X::AbstractDataFrame`: samples to evaluate
- `y::AbstractVector{<:Class}`: target vector
- `selector::AbstractFeaturesSelector`: applied selector

# Keywords

- `frmidx::Integer`: Frame index inside `X`
"""
function buildbitmask(
    X::SoleData.MultiModalDataset,
    selector::AbstractFeaturesSelector,
    frmidx::Integer
)::Tuple{BitVector, BitVector}
    return buildbitmask(SoleData.frame(X, frmidx), selector)
end

# TODO: buildbitmask for MultiModalDataset with supervised selector

function buildbitmask(
    X::AbstractDataFrame,
    selector::AbstractFeaturesSelector
)::BitVector
    idxes = apply(X, selector)
    return _idxes2bm(size(X, 2), idxes)
end

function buildbitmask(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::AbstractFeaturesSelector
)::BitVector
    idxes = apply(X, y, selector)
    return _idxes2bm(size(X, 2), idxes)
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
