module NormalizationExt

using Normalization
using DataTreatments

import Normalization: __mapdims!, fit!, fit, normalize
import Normalization: dimparams, negdims, estimators, dims!, params!

# ---------------------------------------------------------------------------- #
#                 extend fit & normalize to multidim elements                  #
# ---------------------------------------------------------------------------- #
function __mapdims!(z, f, x::AbstractArray{<:AbstractArray}, y)
    @inbounds map!.(f(map(only, y)...), z, x)
end

function fit!(
    T::AbstractNormalization,
    X::AbstractArray{<:AbstractArray{A}};
    dims=Normalization.dims(T)
) where {A}
    eltype(T) == A || throw(TypeError(:fit!, "Normalization", eltype(T), X))

    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(T)) do f
        reshape(
            isnothing(dims) ?
                f(Iterators.flatten(Xs...)) :
                f.(Iterators.flatten.(Xs))
            , nps...
        )
    end
    
    dims!(T, dims)
    params!(T, ps)
    nothing
end

function fit(
    ::Type{𝒯},
    X::AbstractArray{<:AbstractArray{A}};
    dims=nothing
) where {A,T,𝒯<:AbstractNormalization{T}}
    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(𝒯)) do f
        reshape(
            isnothing(dims) ?
                f(Iterators.flatten(Xs...)) :
                f.(Iterators.flatten.(Xs))
            , nps...
        )
    end
    𝒯(dims, ps)
end

function fit(
    ::Type{𝒯},
    X::AbstractArray{<:AbstractArray{A}};
    kwargs...
) where {A,𝒯<:AbstractNormalization}
    fit(𝒯{A}, X; kwargs...)
end

function normalize(X::AbstractArray{<:AbstractArray}, T::AbstractNormalization; kwargs...)
    Y = copy.(X)
    normalize!(Y, T; kwargs...)
    return Y
end

end