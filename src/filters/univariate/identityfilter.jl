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

function score(X::AbstractMatrix, y::Vector{Int64}, selector::IdentityFilter)
    return score(X, selector)
end

function score(X::AbstractArray, selector::IdentityFilter)
    return fill(1.0, size(X, 2))
end
