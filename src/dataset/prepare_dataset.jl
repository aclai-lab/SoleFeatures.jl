# # ---------------------------------------------------------------------------- #
# #                                   utils                                      #
# # ---------------------------------------------------------------------------- #
# check_dataframe_type(df::AbstractDataFrame) = all(col -> eltype(col) <: Union{Real,AbstractArray{<:Real}}, eachcol(df))
# hasnans(X::AbstractDataFrame) = any(x -> x == 1, SoleData.hasnans.(eachcol(X)))

# ---------------------------------------------------------------------------- #
#                              check dimensions                                #
# ---------------------------------------------------------------------------- #
"""
    _check_dimensions(X::DataFrame) -> Int

Internal function.
Check dimensionality of elements in DataFrame columns.
Currently supports only scalar values and time series (1-dimensional arrays).

# Returns
- `Int`: 0 for scalar elements, 1 for 1D array elements

# Throws
- `DimensionMismatch`: If elements have inconsistent dimensions
- `ArgumentError`: If elements have more than 1D
"""
function _check_dimensions(X::DataFrame)
    isempty(X) && return 0
    
    # Get reference dimensions from first element
    first_col = first(eachcol(X))
    ref_dims = ndims(first(first_col))
    
    # Early dimension check
    ref_dims > 1 && throw(ArgumentError("Elements more than 1D are not supported."))
    
    # Check all columns maintain same dimensionality
    all(col -> all(x -> ndims(x) == ref_dims, col), eachcol(X)) ||
        throw(DimensionMismatch("Inconsistent dimensions across elements"))
    
    return ref_dims
end

# ---------------------------------------------------------------------------- #
#                                 treatment                                    #
# ---------------------------------------------------------------------------- #
"""
    _treatment(X::DataFrame, vnames::AbstractVector{String}, treatment::Symbol, 
               features::AbstractVector{<:Base.Callable}, winparams::NamedTuple)

Internal function.
Processes the input DataFrame `X` based on the specified `treatment` type, 
either aggregating or reducing the size of the data. The function applies 
the given `features` to the columns specified by `vnames`, using window 
parameters defined in `winparams`.

# Arguments
- `X::DataFrame`: The input data to be processed.
- `vnames::AbstractVector{String}`: Names of the columns in `X` to be treated.
- `treatment::Symbol`: The type of treatment to apply, either `:aggregate` 
  or `:reducesize`.
- `features::AbstractVector{<:Base.Callable}`: Functions to apply to the 
  specified columns.
- `winparams::NamedTuple`: Parameters defining the windowing strategy, 
  including the type of window function.

# Returns
- `DataFrame`: A new DataFrame with the processed data.

# Throws
- `ArgumentError`: If `winparams` does not contain a valid `type`.
"""
function _treatment(
    X::DataFrame,
    vnames::AbstractVector{String},
    treatment::Symbol,
    features::AbstractVector{<:Base.Callable},
    winparams::NamedTuple
)
    # check parameters
    haskey(winparams, :type) || throw(ArgumentError("winparams must contain a type, $(keys(WIN_PARAMS))"))
    haskey(WIN_PARAMS, winparams.type) || throw(ArgumentError("winparams.type must be one of: $(keys(WIN_PARAMS))"))

    max_interval = maximum(length.(eachrow(X)))
    _wparams = NamedTuple(k => v for (k,v) in pairs(winparams) if k != :type)
    n_intervals = winparams.type(max_interval; _wparams...)

    # Initialize DataFrame
    if treatment == :aggregate        # propositional
        if n_intervals == 1
            valid_X = DataFrame([v => Float64[]
                                 for v in [string(f, "(", v, ")")
                                       for f in features for v in vnames]]
            )
        else
            valid_X = DataFrame([v => Float64[]
                                 for v in [string(f, "(", v, ")w", i)
                                       for f in features for v in vnames
                                       for i in 1:length(n_intervals)]]
            )
        end

    elseif treatment == :reducesize   # modal
        # valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])
        valid_X = DataFrame([name => Vector{Float64}[] for name in vnames])

    elseif treatment == :feature_selection
        if n_intervals == 1
            # valid_X = DataFrame([v => Float64[]
            valid_X = DataFrame([v => Feature[]
                for v in [string(f, "(", v, ")")
                    for f in features for v in vnames]]
            )
        else
            # valid_X = DataFrame([v => Float64[]
            valid_X = DataFrame([v => Feature[]
                for v in [string(f, "(", v, ")w", i)
                    for f in features for v in vnames
                    for i in 1:length(n_intervals)]]
            )
        end
    end

    # Fill DataFrame
    for row in eachrow(X)
        row_intervals = winparams.type(maximum(length.(collect(row))); _wparams...)
        # interval_dif is used in case we encounter a row with less intervals than the maximum
        interval_diff = length(n_intervals) - length(row_intervals)

        if treatment == :aggregate
            push!(valid_X, vcat([
                vcat([f(col[r]) for r in row_intervals],
                    # if interval_diff is positive, fill the rest with NaN
                    fill(NaN, interval_diff)) for col in row, f in features
                ]...)
            )
        elseif treatment == :reducesize
            f = haskey(_wparams, :reducefunc) ? _wparams.reducefunc : mean
            push!(valid_X, [
                vcat([f(col[r]) for r in row_intervals],
                    # if interval_diff is positive, fill the rest with NaN
                    fill(NaN, interval_diff)) for col in row
                ]
            )
        elseif treatment == :feature_selection
            push!(valid_X, vcat([
                vcat([
                    Feature(f(col[r]), vnames[i], Symbol(f), w) for (w, r) in enumerate(row_intervals)],
                    # if interval_diff is positive, fill the rest with NaN
                    fill(NaN, interval_diff)) for (i, col) in enumerate(row), f in features
                ]...)
            )
        end
    end

    return valid_X
end

# # ---------------------------------------------------------------------------- #
# #                                 partitioning                                 #
# # ---------------------------------------------------------------------------- #
# """
#     _partition(y::Union{CategoricalArray, Vector{T}}, train_ratio::Float64, 
#                shuffle::Bool, stratified::Bool, nfolds::Int, rng::AbstractRNG) 
#                where {T<:Union{AbstractString, Number}}

# Partitions the input vector `y` into training and testing indices based on 
# the specified parameters. Supports both stratified and non-stratified 
# partitioning.

# # Arguments
# - `y::Union{CategoricalArray, Vector{T}}`: The target variable to partition.
# - `train_ratio::Float64`: The ratio of data to be used for training in 
#   non-stratified partitioning.
# - `shuffle::Bool`: Whether to shuffle the data before partitioning.
# - `stratified::Bool`: Whether to perform stratified partitioning.
# - `nfolds::Int`: Number of folds for cross-validation in stratified 
#   partitioning.
# - `rng::AbstractRNG`: Random number generator for reproducibility.

# # Returns
# - `Vector{Tuple{Vector{Int}, Vector{Int}}}`: A vector of tuples containing 
#   training and testing indices.

# # Throws
# - `ArgumentError`: If `nfolds` is less than 2 when `stratified` is true.
# """

# function _partition(
#     y::Union{CategoricalArray,Vector{T}},
#     # validation::Bool,
#     train_ratio::Float64,
#     valid_ratio::Float64,
#     shuffle::Bool,
#     stratified::Bool,
#     nfolds::Int,
#     rng::AbstractRNG
# ) where {T<:Union{AbstractString,Number}}
#     if stratified
#         stratified_cv = MLJ.StratifiedCV(; nfolds, shuffle, rng)
#         tt = MLJ.MLJBase.train_test_pairs(stratified_cv, 1:length(y), y)
#         if valid_ratio == 1.0
#             return [TT_indexes(train, eltype(train)[], test) for (train, test) in tt]
#         else
#             tv = collect((MLJ.partition(t[1], train_ratio)..., t[2]) for t in tt)
#             return [TT_indexes(train, valid, test) for (train, valid, test) in tv]
#         end
#     else
#         tt = MLJ.partition(eachindex(y), train_ratio; shuffle, rng)
#         if valid_ratio == 1.0
#             return TT_indexes(tt[1], eltype(tt[1])[], tt[2])
#         else
#             tv = MLJ.partition(tt[1], valid_ratio; shuffle, rng)
#             return TT_indexes(tv[1], tv[2], tt[2])
#         end
#     end
# end

# # ---------------------------------------------------------------------------- #
# #                               prepare dataset                                #
# # ---------------------------------------------------------------------------- #
# """
#     prepare_dataset(X::AbstractDataFrame, y::AbstractVector; algo::Symbol=:classification, 
#                     treatment::Symbol=:aggregate, features::AbstractVector{<:Base.Callable}=DEFAULT_FEATS, 
#                     train_ratio::Float64=0.8, shuffle::Bool=true, stratified::Bool=false, 
#                     nfolds::Int=6, rng::AbstractRNG=Random.TaskLocalRNG(), 
#                     winparams::Union{NamedTuple,Nothing}=nothing, 
#                     vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}},Nothing}=nothing)

# Prepares a dataset for machine learning by processing the input DataFrame `X` and target vector `y`. 
# Supports both classification and regression tasks, with options for data treatment and partitioning.

# # Arguments
# - `X::AbstractDataFrame`: The input data containing features.
# - `y::AbstractVector`: The target variable corresponding to the rows in `X`.
# - `algo::Symbol`: The type of algorithm, either `:classification` or `:regression`.
# - `treatment::Symbol`: The data treatment method, default is `:aggregate`.
# - `features::AbstractVector{<:Base.Callable}`: Functions to apply to the data columns.
# - `train_ratio::Float64`: Ratio of data to be used for training.
# - `shuffle::Bool`: Whether to shuffle data before partitioning.
# - `stratified::Bool`: Whether to use stratified partitioning.
# - `nfolds::Int`: Number of folds for cross-validation.
# - `rng::AbstractRNG`: Random number generator for reproducibility.
# - `winparams::Union{NamedTuple,Nothing}`: Parameters for windowing strategy.
# - `vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}},Nothing}`: Names of the columns in `X`.

# # Returns
# - `SoleXplorer.Dataset`: A dataset object containing processed data and partitioning information.

# # Throws
# - `ArgumentError`: If input parameters are invalid or unsupported column types are encountered.
# """

# function prepare_dataset(
#     X::AbstractDataFrame,
#     y::AbstractVector;
#     # model.config
#     algo::Symbol=:classification,
#     treatment::Symbol=:aggregate,
#     features::AbstractVector{<:Base.Callable}=DEFAULT_FEATS,
#     # validation::Bool=false,
#     # model.preprocess
#     train_ratio::Float64=0.8,
#     valid_ratio::Float64=1.0,
#     shuffle::Bool=true,
#     stratified::Bool=false,
#     nfolds::Int=6,
#     rng::AbstractRNG=Random.TaskLocalRNG(),
#     # model.winparams
#     winparams::Union{NamedTuple,Nothing}=nothing,
#     vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}},Nothing}=nothing,
# )
#     # check parameters
#     check_dataframe_type(X) || throw(ArgumentError("DataFrame must contain only numeric values"))
#     size(X, 1) == length(y) || throw(ArgumentError("Number of rows in DataFrame must match length of class labels"))
#     treatment in AVAIL_TREATMENTS || throw(ArgumentError("Treatment must be one of: $AVAIL_TREATMENTS"))

#     if algo == :regression
#         y isa AbstractVector{<:Number} || throw(ArgumentError("Regression requires a numeric target variable"))
#         y isa AbstractFloat || (y = Float64.(y))
#     elseif algo == :classification
#         y isa AbstractVector{<:AbstractFloat} && throw(ArgumentError("Classification requires a categorical target variable"))
#         y isa CategoricalArray || (y = coerce(y, MLJ.Multiclass))
#     else
#         throw(ArgumentError("Algorithms supported, :regression and :classification"))
#     end

#     if isnothing(vnames)
#         vnames = names(X)
#     else
#         size(X, 2) == length(vnames) || throw(ArgumentError("Number of columns in DataFrame must match length of variable names"))
#         vnames = eltype(vnames) <: Symbol ? string.(vnames) : vnames
#     end

#     hasnans(X) && @warn "DataFrame contains NaN values"

#     column_eltypes = eltype.(eachcol(X))

#     ds_info = DatasetInfo(
#         algo,
#         treatment,
#         features,
#         train_ratio,
#         valid_ratio,
#         shuffle,
#         stratified,
#         nfolds,
#         rng,
#         winparams,
#         vnames,
#         # validation
#     )

#     # case 1: dataframe with numeric columns
#     if all(t -> t <: Number, column_eltypes)
#         return SoleXplorer.Dataset(
#             DataFrame(vnames .=> eachcol(X)), y,
#             # _partition(y, validation, train_ratio, valid_ratio, shuffle, stratified, nfolds, rng),
#             _partition(y, train_ratio, valid_ratio, shuffle, stratified, nfolds, rng),
#             ds_info
#         )
#     # case 2: dataframe with vector-valued columns
#     elseif all(t -> t <: AbstractVector{<:Number}, column_eltypes)
#         return SoleXplorer.Dataset(
#             # if winparams is nothing, then leave the dataframe as it is
#             isnothing(winparams) ? DataFrame(vnames .=> eachcol(X)) : _treatment(X, vnames, treatment, features, winparams), y,
#             # _partition(y, validation, train_ratio, valid_ratio, shuffle, stratified, nfolds, rng),
#             _partition(y, train_ratio, valid_ratio, shuffle, stratified, nfolds, rng),
#             ds_info
#         )
#     else
#         throw(ArgumentError("Column type not yet supported"))
#     end
# end

# function prepare_dataset(
#     X::AbstractDataFrame,
#     y::AbstractVector,
#     model::AbstractModelSet
# )
#     # check if it's needed also validation set
#     # validation = haskey(VALIDATION, model.type) && getproperty(model.params, VALIDATION[model.type][1]) != VALIDATION[model.type][2]
#     # valid_ratio = (validation && model.preprocess.valid_ratio == 1) ? 0.8 : model.preprocess.valid_ratio

#     prepare_dataset(
#         X, y;
#         algo=model.config.algo,
#         treatment=model.config.treatment,
#         features=model.features,
#         # validation,
#         # model.preprocess
#         train_ratio=model.preprocess.train_ratio,
#         valid_ratio=model.preprocess.valid_ratio,
#         shuffle=model.preprocess.shuffle,
#         stratified=model.preprocess.stratified,
#         nfolds=model.preprocess.nfolds,
#         rng=model.preprocess.rng,
#         winparams=model.winparams,
#     )
# end

# # y is not a vector, but a symbol or a string that identifies the column in X
# function prepare_dataset(
#     X::AbstractDataFrame,
#     y::Union{Symbol,AbstractString},
#     args...; kwargs...
# )
#     prepare_dataset(X[!, Not(y)], X[!, y], args...; kwargs...)
# end

# ---------------------------------------------------------------------------- #
#                        feature selection preprocess                          #
# ---------------------------------------------------------------------------- #
"""
    feature_selection_preprocess(
        X::DataFrame;
        vnames::Union{Vector{String}, Vector{Symbol}, Nothing}=nothing,
        features::Union{Vector{<:Base.Callable}, Nothing}=nothing,
        nwindows::Union{Int, Nothing}=nothing
    ) -> DataFrame

Process a DataFrame for feature selection by converting its columns into Feature objects.

# Arguments
- `X::DataFrame`: Input DataFrame containing time series data
- `vnames::Union{Vector{String}, Vector{Symbol}, Nothing}=nothing`: Names for the variables. 
   If nothing, uses DataFrame column names
- `features::Union{Vector{<:Base.Callable}, Nothing}=nothing`: Feature extraction functions. 
   If nothing, uses DEFAULT_FE.features
- `nwindows::Union{Int, Nothing}=nothing`: Number of windows for time series segmentation. 
   If nothing, uses DEFAULT_FE_WINPARAMS

# Returns
- `DataFrame`: A DataFrame where each element is a Feature object containing:
  - value: extracted feature value
  - var: variable name
  - feats: feature extraction function used
  - nwin: window number

# Example
```julia
# Basic usage with default parameters
df = DataFrame(a = [rand(10) for _ in 1:5])
result = feature_selection_preprocess(df)

# Custom features and windows
df = DataFrame(a = [rand(10) for _ in 1:5])
result = feature_selection_preprocess(df,
    features = [mean, std],
    nwindows = 3
)
"""
function feature_selection_preprocess(
    X::DataFrame;
    vnames::Union{Vector{String}, Vector{Symbol}, Nothing}=nothing,
    features::Union{Vector{<:Base.Callable}, Nothing}=nothing,
    nwindows::Union{Int, Nothing}=nothing
)
    # check parameters
    isnothing(vnames) && (vnames = names(X))
    isnothing(features) && (features = DEFAULT_FE.features)
    treatment = :feature_selection
    _ = _check_dimensions(X)
    if !isnothing(nwindows)
        nwindows > 0 || throw(ArgumentError("Number of windows must be positive."))
    end
    winparams = isnothing(nwindows) ? DEFAULT_FE_WINPARAMS : merge(DEFAULT_FE_WINPARAMS, (nwindows = nwindows,))

    _treatment(X, vnames, treatment, features, winparams)
end
