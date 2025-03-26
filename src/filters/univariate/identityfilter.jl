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

function score(X::AbstractMatrix, y::AbstractVector{<:Class}, selector::IdentityFilter)
    return score(X, selector)
end
score(Xdf::AbstractDataFrame, y::AbstractVector{<:Class}, selector::IdentityFilter) = score(Matrix(Xdf), y, selector)

function score(X::AbstractMatrix, selector::IdentityFilter)
    return fill(1.0, size(X, 2))
end
score(Xdf::AbstractDataFrame, selector::IdentityFilter) = score(Matrix(Xdf), selector)
