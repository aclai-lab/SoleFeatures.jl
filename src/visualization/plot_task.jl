abstract type AbstractPlotTask end

params(task::AbstractPlotTask) = task.params

# ---------------------------------------------------------------------------------------- #
#                                      Histogram Plot                                      #
# ---------------------------------------------------------------------------------------- #

struct HistogramTask <: AbstractPlotTask
    vals::Vector{<:Float64}
    params::NamedTuple

    function HistogramTask(vals::AbstractVector{<:Real}; params::NamedTuple=(;))
        new(vals, params)
    end

    function HistogramTask(
        vals::AbstractVector{<:Union{Missing, T}};
        params::NamedTuple=(;)
    ) where {T<:Real}
        cleaned = skipmissing(vals) |> collect
        new(Float64.(cleaned), params)
    end
end

vals(task::HistogramTask) = task.vals

run_task(task::HistogramTask) = Plots.histogram(vals(task); params(task)...)

# ---------------------------------------------------------------------------------------- #
#                                           Bar Plot                                       #
# ---------------------------------------------------------------------------------------- #

struct BarTask <: AbstractPlotTask
    labels::Vector{String}
    vals::Vector{<:Float64}
    params::NamedTuple

    function BarTask(data::AbstractVector; params::NamedTuple=(;))
        counts = countmap(skipmissing(data))

        labels = string.(collect(keys(counts)))
        vals = Float64.(collect(values(counts)))

        return new(labels, vals, params)
    end

    function BarTask(
        labels::AbstractVector,
        vals::AbstractVector{<:Real};
        params::NamedTuple=(;)
    )
        n_labels = length(labels)
        n_vals = length(vals)

        @assert n_labels == n_vals "Mismatch: $n_labels labels vs $n_vals vals"

        return new(string.(labels), Float64.(vals), params)
    end
end

labels(task::BarTask) = task.labels
vals(task::BarTask) = task.vals

run_task(task::BarTask) = Plots.bar(labels(task), vals(task); params(task)...)

# ---------------------------------------------------------------------------------------- #
#                                         Density Plot                                     #
# ---------------------------------------------------------------------------------------- #

struct DensityTask <: AbstractPlotTask
    vals::Vector{Float64}
    params::NamedTuple

    function DensityTask(vals::AbstractVector{<:Real}; params::NamedTuple=(;))
        new(Float64.(vals), params)
    end

    function DensityTask(
        vals::AbstractVector{Union{Missing, T}};
        params::NamedTuple=(;)
    ) where {T<:Real}
        cleaned = skipmissing(vals) |> collect
        new(Float64.(cleaned), params)
    end
end

vals(task::DensityTask) = task.vals

run_task(task::DensityTask) = Plots.density(vals(task); params(task)...)

# ---------------------------------------------------------------------------------------- #
#                                         Boxplot Plot                                     #
# ---------------------------------------------------------------------------------------- #

struct BoxplotTask <: AbstractPlotTask
    vals::Vector{Float64}
    params::NamedTuple

    function BoxplotTask(vals::AbstractVector{<:Real}; params::NamedTuple=(;))
        new(Float64.(vals), params)
    end

    function BoxplotTask(
        vals::AbstractVector{<:Union{Missing, T}};
        params::NamedTuple=(;)
    ) where {T<:Real}
        cleaned = collect(skipmissing(vals))
        new(Float64.(cleaned), params)
    end
end

vals(task::BoxplotTask) = task.vals

run_task(task::BoxplotTask) = Plots.boxplot(vals(task); params(task)...)

# ---------------------------------------------------------------------------------------- #
#                                         Violin Plot                                      #
# ---------------------------------------------------------------------------------------- #

struct ViolinTask <: AbstractPlotTask
    vals::Vector{Float64}
    params::NamedTuple

    function ViolinTask(vals::AbstractVector{<:Real}; params::NamedTuple=(;))
        new(Float64.(vals), params)
    end

    function ViolinTask(
        vals::AbstractVector{<:Union{Missing, T}};
        params::NamedTuple=(;)
    ) where {T<:Real}
        cleaned = collect(skipmissing(vals))
        new(Float64.(cleaned), params)
    end
end

vals(task::ViolinTask) = task.vals

run_task(task::ViolinTask) = Plots.violin(vals(task); params(task)...)

# ---------------------------------------------------------------------------------------- #
#                                          QQ-Plot                                         #
# ---------------------------------------------------------------------------------------- #

struct QQplotTask <: AbstractPlotTask
    x::Vector{Float64}
    y::Vector{Float64}
    params::NamedTuple

    function QQplotTask(
        x::AbstractVector{<:Real},
        y::AbstractVector{<:Real};
        params::NamedTuple=(;)
    )
        new(Float64.(x), Float64.(y), params)
    end

    function QQplotTask(
        x::AbstractVector{<:Union{Missing, T}},
        y::AbstractVector{<:Union{Missing, T}};
        params::NamedTuple=(;)
    ) where {T<:Real}

        @assert length(x) == length(y) "x and y must have the same length"

        mask = .!ismissing.(x) .& .!ismissing.(y)

        cleaned_x = Float64.(x[mask])
        cleaned_y = Float64.(y[mask])

        new(cleaned_x, cleaned_y, params)
    end
end

x(task::QQplotTask) = task.x
y(task::QQplotTask) = task.y

run_task(task::QQplotTask) = StatsPlots.qqplot(x(task), y(task); params(task)...)

# ---------------------------------------------------------------------------------------- #
#                                     Scatter Plot                                         #
# ---------------------------------------------------------------------------------------- #

struct ScatterTask <: AbstractPlotTask
    x::AbstractVector
    y::Union{AbstractVector, Nothing}
    params::NamedTuple

    function ScatterTask(
        x::AbstractVector;
        params::NamedTuple=(;)
    )
        new(x, nothing, params)
    end

    function ScatterTask(
        x::AbstractVector,
        y::AbstractVector;
        params::NamedTuple=(;)
    )
        @assert length(x) == length(y) "x and y must have the same length"

        mask = .!ismissing.(x) .& .!ismissing.(y)

        cleaned_x = x[mask]
        cleaned_y = y[mask]

        new(cleaned_x, cleaned_y, params)
    end
end

x(task::ScatterTask) = task.x
y(task::ScatterTask) = task.y

function run_task(task::ScatterTask)
    if isnothing(y(task))
        return Plots.scatter(x(task); params(task)...)
    end

    return Plots.scatter(x(task), y(task); params(task)...)
end

# ---------------------------------------------------------------------------------------- #
#                                         Density Plot                                     #
# ---------------------------------------------------------------------------------------- #

struct MultiTask
    tasks::AbstractVector{<:AbstractPlotTask}
end

tasks(multitask::MultiTask) = multitask.tasks

function run_task(
    multitask::MultiTask;
    layout::Tuple{Int, Int}=(1, length(tasks(multitask)))
)
    plots = [run_task(task) for task in tasks(multitask)]
    Plots.plot(plots...; layout=layout)
end
