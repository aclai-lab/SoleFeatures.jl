# ---------------------------------------------------------------------------- #
#                                 identity filter                              #
# ---------------------------------------------------------------------------- #
"""
    IdentityFilter{T <: AbstractLimiter} <: AbstractIdentityFilter{T}

A unsupervised and supervised univariate feature selection filter that always
return the same score (1.0) for each feature.

# Fields
- `limiter::T`: A limiter that defines the selection criterion to be applied to scores.
"""
struct IdentityFilter{T <: AbstractLimiter} <: AbstractIdentityFilter{T}
    limiter::T
    # TODO parameters

    function IdentityFilter()
        new{IdentityLimiter}(IdentityLimiter())
    end
end

is_supervised(::AbstractIdentityFilter) = true
is_unsupervised(::AbstractIdentityFilter) = true

"""
    score(selector, X, y)
    score(selector, X)

Return a vector of scores (1.0) for each feature in `X`.

# Arguments
- `selector::IdentityFilter`: Instance of IdentityFilter
- `X::AbstractArray{<:Real}`: Feature matrix (n_samples × n_features)
- `y::AbstractVector` (optional): Target labels (not used)

# Example
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
scores = SoleFeatures.score(IdentityFilter(), X, y)
```
"""
function score(selector::IdentityFilter, X::AbstractMatrix, y::Vector{Int64})
    return score(selector, X)
end

function score(selector::IdentityFilter, X::AbstractArray)
    return fill(1.0, size(X, 2))
end
