# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
struct RtestFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function RtestFilter(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
        rank, score = _f_statistic_regress(X, y)
        new{T,RegressionTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                            r_statistic regression                            #
# ---------------------------------------------------------------------------- #
function _r_statistic_regress(X::AbstractArray{T}, y::AbstractVector{<:AbstractFloat}) where {T<:Real}
    n_samples = size(X, 1)
    
    y_centered = y .- mean(y)
    X_means = mean(X, dims=1)
    X_squared_sum = sum(X.^2, dims=1)
    X_norms = sqrt.(X_squared_sum .- n_samples .* X_means.^2)
    
    f_result = vec((y_centered' * X) ./ X_norms ./ LinearAlgebra.norm(y_centered))

    nan_mask = isnan.(f_result)
    f_result[nan_mask] .= 0.0
    
    return sortperm(f_result, rev=true), f_result
end

