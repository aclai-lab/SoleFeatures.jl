"""
    Summary Statistics for discrete values
"""
function _summary_stats(vals::AbstractVector{<:Union{AbstractString, Symbol}})
    freq = StatsBase.countmap(vals)
    n = length(vals)
    p = collect(values(freq)) ./ n

    return Dict(
        "absolute_frequency" => freq,
        "relative_frequency" => Dict(k => v / n for (k, v) in freq),
        "mode"  => StatsBase.mode(vals),
        "entropy" => StatsBase.entropy(p)
    )
end

"""
    Summary statistics for continuous values
"""
function _summary_stats(vals::AbstractVector{<:Real})
    return Dict(
        "mean" => Statistics.mean(vals),
        "median" => Statistics.median(vals),
        "mode"  => StatsBase.mode(vals),
        "variance" => Statistics.var(vals),
        "std" => Statistics.std(vals),
        "skewness" => StatsBase.skewness(vals),
        "kurtosis" => StatsBase.kurtosis(vals),
        "minimum" => minimum(vals),
        "maximum" => maximum(vals)
    )
end

"""
    Summary statistics to clean data from missing values
"""
function summary_stats(vals::AbstractVector)
    cleaned_vals = collect(skipmissing(vals))
    length(cleaned_vals) == 0 && error("No non-missing values to compute statistics.")

    return _summary_stats(cleaned_vals)
end


# ---------------------------------------------------------------------------------------- #
#                                 Summary statistics table                                 #
# ---------------------------------------------------------------------------------------- #

function summary_table(
    vals::AbstractMatrix,
    col_names::Vector{String},
    args...;
    kwargs...
)
    return summary_table(DataFrame(vals, Symbol.(col_names)), args...; kwargs...)
end

function summary_table(
    df::DataFrames.DataFrame;
    addnmissing::Bool = true,
    digits::Int = 1,
    varnames::Union{Nothing, AbstractDict{Symbol, <:Any}} = nothing,
    kwargs...
)
    return TableOne.tableone(
        df;
        addnmissing = addnmissing,
        digits = digits,
        varnames = varnames
    )
end

function summary_table(
    df::DataFrames.DataFrame,
    strata::Symbol,
    vars::Union{Nothing, Vector{Symbol}} = nothing;
    binvars::Vector{Symbol} = Symbol[],
    catvars::Vector{Symbol} = Symbol[],
    npvars::Vector{Symbol} = Symbol[],
    paramvars::Vector{Symbol} = Symbol[],
    addnmissing::Bool = true,
    addtestname::Bool = false,
    addtotal::Bool = false,
    binvardisplay::Union{Nothing, Dict{Symbol, Any}} = nothing,
    digits::Int = 1,
    includemissingintotal::Bool = false,
    pdigits::Int = 3,
    pvalues::Bool = false,
    varnames::Union{Nothing, AbstractDict{Symbol, <:Any}} = nothing,
    kwargs...
)
    vars = isnothing(vars) ? Symbol.(names(df)) : Symbol.(vars)

    return TableOne.tableone(
        df,
        strata,
        vars;
        binvars = binvars,
        catvars = catvars,
        npvars = npvars,
        paramvars = paramvars,
        addnmissing = addnmissing,
        addtestname = addtestname,
        addtotal = addtotal,
        binvardisplay = binvardisplay,
        digits = digits,
        includemissingintotal = includemissingintotal,
        pdigits = pdigits,
        pvalues = pvalues,
        varnames = varnames,
        kwargs...
    )
end
