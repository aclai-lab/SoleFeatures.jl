struct SupLaplacianScoreFilter{T <: AbstractLimiter} <: AbstractSupLaplacianScoreFilter{T}
    limiter::T,
    # parameters
    w_metric::String,
    w_weight_mode::String,
    k::Integer,
    t::Real
end

limiter(selector::LaplacianScoreFilter) = selector.limiter
w_metric(selector::LaplacianScoreFilter) = selector.w_metric
w_weight_mode(selector::LaplacianScoreFilter) = selector.w_weight_mode
k(selector::LaplacianScoreFilter) = selector.k
t(selector::LaplacianScoreFilter) = selector.t

# ========================================================================================
# CUSTOM CONSTRUCTORS

# ========================================================================================
# Shared apply functions

function apply(X::AbstractDataFrame, selector::VarianceFilter{<:AbstractLimiter})
    # X =
end
