# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    PearsonCorFilter{T <: AbstractLimiter} <: AbstractPearsonCorFilter{T}

A supervised univariate feature selection filter that intends to determine the 
correlation between each continuous feature and a discrete target variable. The 
linear relationship is measured by the Pearson's correlation coefficient, whose 
score ranges from -1 (negative correlation) to 1 (positive correlation).

# Fields
- `limiter::T`: A limiter that defines the selection criterion to be applied to scores.

# Implementation Details
This implementation follows the scikit-learn estimator and uses the Ross et al. 
estimator for mutual information between continuous and discrete variables.
"""
struct PearsonCorFilter{T <: AbstractLimiter} <: AbstractPearsonCorFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractPearsonCorFilter) = true
is_unsupervised(::AbstractPearsonCorFilter) = false

# ---------------------------------------------------------------------------- #
#                               pearson filter                                 #
# ---------------------------------------------------------------------------- #
"""
    score(X, y, selector)

Compute Pearson correlation between each feature in `X` and the discrete target `y`.

# Arguments
- `X::AbstractArray{<:Real}`: Feature matrix (n_samples × n_features)
- `y::AbstractVector`: Discrete target labels
- `selector::PearsonCorFilter`: 

# Algorithm
1. Check that all columns in `X` are of type `Real`
2. For each feature, compute Pearson correlation coefficient with target `y`

# Example
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
scores = SoleFeatures.score(X, y, PearsonCorFilter(RankingLimiter(3, false)))
```
"""
function score(
    selector::PearsonCorFilter,
    X::AbstractArray{T},
    y::Vector{Int64}
)::Vector{Float64} where {T<:Float64}
    coltypes = eltype.(eachcol(X))
    uncalcidxes = findall(==(false), coltypes .<: Real)
    if (!isempty(uncalcidxes))
        throw(DomainError("Columns must be subtype of Real.\n
        The following column indices are not handable: $(uncalcidxes)"))
    end
    scores = cor.(eachcol(X), [y])
    return scores
end

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
get_pearson_cor_identity() =  PearsonCorFilter(IdentityLimiter())
get_pearson_cor_threshold(threshold::Real, ordf::Function) =  PearsonCorFilter(ThresholdLimiter(threshold, ordf))
get_pearson_cor_ranking(nbest::Integer, rev::Bool=false) =  PearsonCorFilter(RankingLimiter(nbest, rev))
get_pearson_cor_percentage(perc::Float64, rev::Bool=true) =  PearsonCorFilter(PercentageLimiter(perc, rev))