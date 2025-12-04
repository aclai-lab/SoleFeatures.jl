# ---------------------------------------------------------------------------- #
#                              variance filter                                 #
# ---------------------------------------------------------------------------- #
"""
    VarianceFilter{T <: AbstractLimiter} <: AbstractVarianceFilter{T}

A unsupervised univariate feature selection filter that return the variance of 
each feature.

# Fields
- `limiter::T`: A limiter that defines the selection criterion to be applied to scores.
- `corrected::Bool`: A boolean indicating whether to use the corrected sample variance.
- `mean::Union{Nothing, Float64}`: An optional precomputed mean value for variance calculation.
"""
struct VarianceFilter{T<:AbstractLimiter} <: AbstractVarianceFilter{T}
    limiter::T

    # parameters
    corrected::Bool
    mean::Union{Nothing, Float64}

    function VarianceFilter(limiter::T; 
        corrected::Bool=true, mean::Union{Nothing, Float64}=nothing,
    ) where {T<:AbstractLimiter}
        new{T}(limiter, corrected, mean)
    end
end

is_supervised(::AbstractVarianceFilter) = false
is_unsupervised(::AbstractVarianceFilter) = true

"""
    score(selector, X)

Compute variance scores for each feature in `X`.

# Arguments
- `selector::VarianceFilter`: Instance of VarianceFilter
- `X::AbstractArray`: Feature matrix (n_samples × n_features)

# Example
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
scores = SoleFeatures.score(VarianceFilter(RankingLimiter(3, false)), X)
```
"""
function score(selector::VarianceFilter, X::AbstractArray)::Vector{Float64}
    return Statistics.var.(eachcol(X); corrected=selector.corrected,
                                       mean=selector.mean)
end

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
get_variance_identity(; 
    corrected::Bool=true, mean::Union{Nothing, Float64}=nothing) =  VarianceFilter(
        IdentityLimiter(); corrected=corrected, mean=mean,
)
get_variance_threshold(
    threshold::Real, ordf::Function=≥; 
    corrected::Bool=true, mean::Union{Nothing, Float64}=nothing) =  VarianceFilter(
        ThresholdLimiter(threshold, ordf); corrected=corrected, mean=mean,
)
get_variance_ranking(
    nbest::Integer, rev::Bool=true; 
    corrected::Bool=true, mean::Union{Nothing, Float64}=nothing) =  VarianceFilter(
        RankingLimiter(nbest, rev); corrected=corrected, mean=mean,
)
get_variance_percentage(
    perc::Float64, rev::Bool=true; 
    corrected::Bool=true, mean::Union{Nothing, Float64}=nothing) =  VarianceFilter(
        PercentageLimiter(perc, rev); corrected=corrected, mean=mean,
)