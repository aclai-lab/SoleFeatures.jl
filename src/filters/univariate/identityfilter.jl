# ---------------------------------------------------------------------------- #
#                                 identity filter                              #
# ---------------------------------------------------------------------------- #
struct IdentityFilter{T <: AbstractLimiter} <: AbstractIdentityFilter{T}
    limiter::T
    # TODO parameters

    function IdentityFilter()
        new{IdentityLimiter}(IdentityLimiter())
    end
end

is_supervised(::AbstractIdentityFilter) = true
is_unsupervised(::AbstractIdentityFilter) = true

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::IdentityFilter
)
    return score(X, selector)
end

function score(
    X::AbstractMatrix,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::IdentityFilter
)
    return score(X, selector)
end

function score(
    X::AbstractDataFrame,
    selector::IdentityFilter
)
    return fill(1.0, ncol(X))
end

function score(
    X::AbstractMatrix,
    selector::IdentityFilter
)
    return fill(1.0, size(X, 2))
end