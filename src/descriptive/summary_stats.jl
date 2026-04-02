"""
    _summary_stats(vals::AbstractVector{<:Union{AbstractString, Symbol}})

Summary of main statistics for categorical values, including absolute and relative
frequencies, mode, and entropy.
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
    _summary_stats(vals::AbstractVector{<:Real})

Summary of main statistics for numerical values, including mean, median, mode, variance,
standard deviation, skewness, kurtosis, minimum, and maximum.
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
    summary_stats(vals::AbstractVector)

Summary of main statistics for a vector of values, automatically handling both categorical
and numerical data. Missing values are ignored in the computation.
"""
function summary_stats(vals::AbstractVector)
    cleaned_vals = collect(skipmissing(vals))
    length(cleaned_vals) == 0 && error("No non-missing values to compute statistics.")

    return _summary_stats(cleaned_vals)
end

# ---------------------------------------------------------------------------------------- #
#                                 Summary statistics table                                 #
# ---------------------------------------------------------------------------------------- #

"""
    summary_table(vals::AbstractMatrix,col_names::Vector{String},args...;kwargs...)

Convenience function to create a summary statistics table from a matrix of values and
corresponding column names. This function will convert the matrix into a DataFrame and then
call the appropriate `summary_table` method to generate the summary statistics table. The
`args...` and `kwargs...` are passed to the underlying `summary_table` function that
handles DataFrames, allowing for flexible customization of the summary table output.
"""
function summary_table(
    vals::AbstractMatrix,
    col_names::Vector{String},
    args...;
    kwargs...
)
    return summary_table(DataFrame(vals, Symbol.(col_names)), args...; kwargs...)
end

"""
    summary_table(df::DataFrames.DataFrame; kwargs...)

Convenience function to create a summary statistics table from a DataFrame. This function
calls the `TableOne.tableone` function from the TableOne package, which provides a
comprehensive summary of the dataset, including counts, percentages, means, medians,
and more. The `kwargs...` are passed directly to the `tableone` function, allowing for
flexible customization of the summary table output, such as adding missing value counts,
specifying variable names, and controlling the number of digits displayed.
"""
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

"""
    summary_table(df::DataFrames.DataFrame, strata::Symbol, args...; kwargs...)

Convenience function to create a stratified summary statistics table from a DataFrame. This
function calls the `TableOne.tableone` function from the TableOne package, which provides a
comprehensive summary of the dataset stratified by a specified variable (strata). The
`args...` and `kwargs...` are passed directly to the `tableone` function, allowing for
flexible customization of the summary table output, such as adding missing value counts,
specifying variable names, and controlling the number of digits displayed. This function is
particularly useful for comparing groups within the dataset based on the strata variable.
"""
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

# ---------------------------------------------------------------------------------------- #
#                                  DataTreatments Interface                                #
# ---------------------------------------------------------------------------------------- #

function summary_stats(dt::DataTreatments.DataTreatment)
    dataset, column_names = DataTreatments.get_tabular(dt)
    return Dict(
        col => summary_stats(dataset[:, i]) for (i, col) in enumerate(column_names)
    )
end

function summary_table(dt::DataTreatments.DataTreatment, args...; kwargs...)
    dataset, column_names = DataTreatments.get_tabular(dt)
    return summary_table(dataset, column_names, args...; kwargs...)
end
