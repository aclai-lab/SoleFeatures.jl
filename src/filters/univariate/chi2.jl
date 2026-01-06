# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    Chi2Filter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}

Compute the chi-squared statistic between each non-negative feature column of `X`
and the class labels `y`.

This score can be used to select the `n_features` with the highest chi-squared
statistic from `X`, which must contain only **non-negative feature values**
(e.g. booleans, counts, or binned continuous features). Labels `y` are expected
to be integers or will be level-encoded if categorical.


If some of your features are continuous, you need to bin them, for
example by using a discretization method.

Recall that the chi-square test measures dependence between stochastic
variables, so using this function "weeds out" the features that are the
most likely to be independent of class and therefore irrelevant for
classification.

# Arguments
- `X::AbstractArray{T}`: Sample vectors of shape (n_samples, n_features) with non-negative values
- `y::AbstractVector`: Target vector (class labels) of shape (n_samples,)

# Fields
- `rank::Vector{Int64}`: Feature indices sorted by chi-squared statistic (descending)
- `score::Vector{Float64}`: Chi-squared statistic for each feature

# Examples
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
chi2stats = Chi2Filter(X, y)
```
"""
struct Chi2Filter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function Chi2Filter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
        # check for non-negative values
        any(X .< 0) && throw(ArgumentError("Input X must be non-negative."))
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))

        rank, score = _chi2(X, y)
        new{eltype(score),ClassificationTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                          chi-squared classifier                              #
# ---------------------------------------------------------------------------- #
function _chi2(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
    classes  = unique(y)
    y_mask   = y .== permutedims(classes)
    @show y_mask'
    @show X
    observed = y_mask' * X

    # handle binary classification case
    length(classes) == 1 && (y_mask = hcat(1 .- y_mask, y_mask))

    class_prob = mean(y_mask, dims=1)
    fcount     = sum(X, dims=1)
    expected   = class_prob' * fcount

    chi2stats, _ = _chi2stats(observed, expected)

    return sortperm(chi2stats, rev=true), chi2stats
end

function _chi2stats(observed::AbstractArray, expected::AbstractArray)
    # compute chi-squared statistic for each feature using vectorized operations
    chi2stats = [sum(let e = expected[i, j]; e > 0 ? (observed[i, j] - e)^2 / e : 0.0 end 
        for i in axes(observed, 1)) 
        for j in axes(observed, 2)]
    
    # compute p-values using chi-squared distribution
    df      = length(axes(observed, 1)) - 1
    pvalues = Distributions.ccdf.(Chisq(df), chi2stats)
    
    return chi2stats, pvalues
end

