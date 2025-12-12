# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    KSTestFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}

Kolmogorov–Smirnov test is a nonparametric test of the equality of continuous or discontinuous,
one-dimensional probability distributions.
It can be used to test whether a sample came from a given reference
probability distribution(one-sample K–S test), or to test whether or not two samples
came from the same distribution (two-sample K–S test).

"""
struct KSTestFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMrMrFilter) = true
is_unsupervised(::AbstractMrMrFilter) = false

# ---------------------------------------------------------------------------- #
#                              kolmogorov_smirnov                              #
# ---------------------------------------------------------------------------- #
function _kolmogorov_smirnov(x::AbstractVector{T}, y::AbstractVector;)::T where {T<:Real}
    classes = unique(y)
    scores  = T[]
    
    for c in classes
        x_in_group  = x[y .== c]
        x_out_group = x[y .!= c]
        
        if length(x_in_group) > 0 && length(x_out_group) > 0
            ks_stat = HypothesisTests.ApproximateTwoSampleKSTest(x_in_group, x_out_group).δ
            push!(scores, ks_stat)
        end
    end
    
    return isempty(scores) ? 0.0 : mean(scores)
end

function _kolmogorov_smirnov(X::AbstractArray{T}, y::AbstractVector)::Vector{T} where {T<:Real}
    nclasses = size(X,2)
    ks_result = Vector{T}(undef, nclasses)
    if nclasses > 10
        Threads.@threads for i in axes(X, 2)
            ks_result[i] = _kolmogorov_smirnov(X[:,i], y)
        end
    else
        for i in axes(X, 2)
            ks_result[i] = _kolmogorov_smirnov(X[:,i], y)
        end
    end
    return ks_result
end

kolmogorov_smirnov()::Function = x, y -> _kolmogorov_smirnov(x, y)
kolmogorov_smirnov(X::AbstractArray, y::AbstractVector) = _kolmogorov_smirnov(X, y)

# ---------------------------------------------------------------------------- #
#                                    score                                     #
# ---------------------------------------------------------------------------- #
function score(
    ::KSTestFilter,
    X     :: AbstractArray{T},
    y     :: AbstractVector
)::Vector{T} where {T<:Real}
    y isa AbstractVector{<:Int} || (y=CategoricalArrays.levelcode.(y))
    return kolmogorov_smirnov(X, yx)
end