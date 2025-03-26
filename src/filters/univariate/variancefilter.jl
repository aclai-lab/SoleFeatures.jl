# ---------------------------------------------------------------------------- #
#                              variance filter                                 #
# ---------------------------------------------------------------------------- #
struct VarianceFilter{T<:AbstractLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractVarianceFilter) = false
is_unsupervised(::AbstractVarianceFilter) = true

function score(X::AbstractMatrix, selector::VarianceFilter)::Vector{Float64}
    # sum is scaled with n-1
    # var(itr; corrected::Bool=true, mean=nothing[, dims])
    return var.(eachcol(X))
end
score(Xdf::AbstractDataFrame, selector::VarianceFilter)::Vector{Float64} = score(Matrix(Xdf), selector)

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, ≥))
