# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
struct FtestFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilterBased{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function FtestFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
        y isa AbstractVector{<:Int} || (y=CategoricalArrays.levelcode.(y))
        rank, score = _f_statistic(X, y)
        new{T,ClassificationTask,Supervised,Univariate}(rank, score)
    end

    function FtestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
        rank, score = _f_statistic(X, y)
        new{T,RegressionTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                                 f_statistic                                  #
# ---------------------------------------------------------------------------- #
function _f_statistic(x::AbstractVector{T}, y::AbstractVector) where {T<:Real}
    # one-way ANOVA F-statistic
    classes = unique(y)
    nclasses, n = length(classes), length(x)

    class_sqr    = Vector{T}(undef, nclasses)
    @inbounds for (i, c) in enumerate(classes)
        mask = y .== c
        class_sqr[i]    = sum(@view x[mask])^2 / sum(mask)
    end

    grand_sqr  = sum(x)^2 / n

    ss_between = sum(class_sqr[i] for i in 1:nclasses) - grand_sqr
    ss_within  = sum(x.^2) - grand_sqr - ss_between

    ms_between = ss_between / (nclasses - 1)
    ms_within = ss_within / (n - nclasses)
    
    return iszero(ms_within) ? zero(T) : ms_between / ms_within
end

function _f_statistic(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
    nclasses = size(X,2)
    f_result = Vector{T}(undef, nclasses)
    if nclasses > 10
        Threads.@threads for i in axes(X, 2)
            f_result[i] = _f_statistic(X[:,i], y)
        end
    else
        for i in axes(X, 2)
            f_result[i] = _f_statistic(X[:,i], y)
        end
    end
    return sortperm(f_result, rev=true), f_result
end
