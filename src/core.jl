# =========================================================================================
# getters

function limiter(selector::AbstractFeaturesSelector)
    !hasproperty(selector, :limiter) && throw(ErrorException("`selector` struct not contain `limiter` field"))
    return selector.limiter
end

# =========================================================================================
# functions

"""
    transform!(X, args..; kwargs...)
    transform!(X, y, args..; kwargs...)

Removes from provided samples attributes indicated by bitmask or selector

# Arguments

- `X::AbstractDataFrame|MultiFrameDataset`: samples to evaluate
- `y::AbstractVector`: target vector
- `bm::BitVector`: Vector of bit containing which attributes are suitable(1) or not(0)
- `selector::AbstractFeaturesSelector`: Selector

# Keywords

- `frmidx::Integer`: Frame index inside `X`
"""
function transform!(X::AbstractDataFrame, bm::BitVector)
    ncol(X) != length(bm) && throw(DimensionMismatch(""))
    return select!(X, findall(bm))
end

function transform!(X::AbstractDataFrame, selector::AbstractFeaturesSelector)
    return transform!(X, buildbitmask(X, selector))
end

function transform!(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::AbstractFeaturesSelector
)
    return transform!(X, buildbitmask(X, y, selector))
end

function transform!(
    X::SoleBase.AbstractMultiFrameDataset,
    bm::BitVector;
    frmidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(frmidx))
        nattributes(X) != length(bm) && throw(DimensionMismatch(""))
        return SoleBase.SoleDataset.dropattributes!(X, findall(!, bm))
    else
        nattributes(X, frmidx) != length(bm) && thow(DimensionMismatch(""))
        return SoleBase.SoleDataset.dropattributes!(X, frmidx, findall(!, bm))
    end
end

function transform!(
    X::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector;
    frmidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(frmidx))
        bm = buildbitmask(SoleBase.SoleDataset.data(X), selector)
    else
        bm = buildbitmask(SoleBase.frame(X, frmidx), selector)
    end
    return transform!(X, bm; frmidx=frmidx)
end

# TODO: transform! for MultiFrameDataset with supervised selector

transform(X::AbstractDataFrame, args...; kwargs...) = transform!(deepcopy(X), args...; kwargs...)
transform(X::SoleBase.AbstractMultiFrameDataset, args...; kwargs...) = transform!(deepcopy(X), args...; kwargs...)
(s::AbstractFeaturesSelector)(X, args; kwargs...) = transform(X, args..., kwargs...)

"""
    buildbitmask(X, selector)
    buildbitmask(X, y, selector)
    buildbitmask(X, frmidx, selector)

return a bitmask containing selected attributes from selector.
True values indicate selected attribute index

# Arguments

- `X::AbstractDataFrame`: samples to evaluate
- `y::AbstractVector{<:Union{String, Symbol}}`: target vector
- `selector::AbstractFeaturesSelector`: applied selector

# Keywords

- `frmidx::Integer`: Frame index inside `X`
"""
function buildbitmask(
    X::SoleBase.MultiFrameDataset,
    frmidx::Integer,
    selector::AbstractFeaturesSelector
)::Tuple{BitVector, BitVector}
    frbm = buildbitmask(SoleBase.frame(X, frmidx), selector) # frame bitmasks
    bm = _fr_bm2mfd_bm(X, frmidx, frbm)
    return bm, frbm
end

# TODO: buildbitmask for MultiFrameDataset with supervised selector

function buildbitmask(
    X::AbstractDataFrame,
    selector::AbstractFeaturesSelector
)::BitVector
    idxes = apply(X, selector)
    return _idxes2bm(size(X, 2), idxes)
end

function buildbitmask(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
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
