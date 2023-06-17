struct SupLaplacianScoreFilter{T<:AbstractLimiter} <: AbstractSupLaplacianScore{T}
    limiter::T
    metric::Symbol
    weightmode::Symbol
    nneighbors::Int

    function SupLaplacianScoreFilter(
        limiter::T,
        metric::Symbol,
        weightmode::Symbol,
        nneighbors::Int
    ) where {T<:AbstractLimiter}
        return new{T}(limiter, metric, weightmode, nneighbors)
    end

    function SupLaplacianScoreFilter(limiter::T) where {T<:AbstractLimiter}
        return new{T}(limiter, :euclidean, :cosine, 5)
    end
end

is_supervised(::AbstractSupLaplacianScore) = true

metric(s::SupLaplacianScoreFilter) = s.metric
weightmode(s::SupLaplacianScoreFilter) = s.weightmode
nneighbors(s::SupLaplacianScoreFilter) = s.nneighbors

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::SupLaplacianScoreFilter
)::Vector{Float64}
    m = Matrix(X)
    w = construct_w.construct_W(
        Matrix(m);
        y = y,
        metric = string(metric(selector)),
        neighbor_mode = "supervised",
        weight_mode = string(weightmode(selector)),
        k = nneighbors(selector)
    )
    score = lap_score.lap_score(m; W=w)
    return replace!(score, NaN => Inf)
end
