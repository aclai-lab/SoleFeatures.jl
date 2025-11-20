# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    Chi2Filter{T<:AbstractLimiter} <: AbstractChi2Filter{T}

A supervised univariate feature selection filter that computes chi-squared (χ²) 
statistics between non-negative features and categorical target classes.

# Fields
- `limiter::T`: A limiter to select top-k features (ranking) or threshold-based selection
"""
struct Chi2Filter{T<:AbstractLimiter} <: AbstractChi2Filter{T}
    limiter::T
    # parameters
end

is_supervised(::AbstractChi2Filter)   = true
is_unsupervised(::AbstractChi2Filter) = false

Chi2Ranking(nbest) = Chi2Filter(RankingLimiter(nbest, false))
Chi2Threshold(; alpha=0.05) = Chi2Filter(ThresholdLimiter(alpha, ≤))

# ---------------------------------------------------------------------------- #
#                                 chi-squared                                  #
# ---------------------------------------------------------------------------- #
# this filter was tested against scikit learn and matlab

"""
    chi2(X::AbstractArray{T}, y::AbstractVector) where {T<:Float64}

Compute chi-squared stats between each non-negative feature and class.

This score can be used to select the `n_features` features with the
highest values for the test chi-squared statistic from X, which must
contain only **non-negative integer feature values** such as booleans or frequencies
(e.g., term counts in document classification), relative to the classes.

If some of your features are continuous, you need to bin them, for
example by using a discretization method.

Recall that the chi-square test measures dependence between stochastic
variables, so using this function "weeds out" the features that are the
most likely to be independent of class and therefore irrelevant for
classification.

# Arguments
- `X::AbstractMatrix`: Sample vectors of shape (n_samples, n_features)
- `y::AbstractVector`: Target vector (class labels) of shape (n_samples,)

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: A tuple containing:
  - Chi2 statistics for each feature
  - P-values for each feature

# Examples
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
chi2stats, pvalues = chi2(X, y)
```
"""
function chi2(
    X::AbstractArray{T},
    y::AbstractVector
)::Tuple{Vector{T}, Vector{T}} where {T<:Float64}
    # check for non-negative values
    any(X .< 0) && throw(ArgumentError("Input X must be non-negative."))
    
    classes  = unique(y)
    y_mask   = y .== permutedims(classes)
    observed = y_mask' * X

    # handle binary classification case
    length(classes) == 1 && (y_mask = hcat(1 .- y_mask, y_mask))

    class_prob = mean(y_mask, dims=1)
    fcount     = sum(X, dims=1)
    expected   = class_prob' * fcount

    return _chi2(observed, expected)
end
chi2(X::AbstractArray{<:Real}, args...) = chi2(Float64.(X), args...)

function _chi2(
    observed::AbstractMatrix{T},
    expected::AbstractMatrix{T}
)::Tuple{Vector{T}, Vector{T}} where {T<:Float64}
    # compute chi-squared statistic for each feature using vectorized operations
    chi2stats = [sum(let e = expected[i, j]; e > 0 ? (observed[i, j] - e)^2 / e : 0.0 end 
        for i in axes(observed, 1)) 
        for j in axes(observed, 2)]
    
    # compute p-values using chi-squared distribution
    df      = length(axes(observed, 1)) - 1
    pvalues = Distributions.ccdf.(Chisq(df), chi2stats)
    
    return chi2stats, pvalues
end

# ---------------------------------------------------------------------------- #
#                                    score                                     #
# ---------------------------------------------------------------------------- #
function score(
    ::Chi2Filter,
    X::AbstractArray{T},
    y::AbstractVector
)::Vector{Float64} where {T<:Real}
    _, pvalues = chi2(X, y)
    return pvalues
end
