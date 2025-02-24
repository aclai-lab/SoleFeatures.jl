# ---------------------------------------------------------------------------- #
#                                    dataset                                   #
# ---------------------------------------------------------------------------- #
# const DEFAULT_PREPROC = (
#     train_ratio = 0.8,
#     valid_ratio = 1.0,
#     shuffle     = true,
#     stratified  = false,
#     nfolds      = 6,
#     rng         = TaskLocalRNG()
# )

const DEFAULT_FE = (
    features = catch9,
)
const DEFAULT_FE_WINPARAMS = (
    type = adaptivewindow,
    nwindows = 10,
    relative_overlap = 0.2
)

# const AVAIL_WINS       = (movingwindow, wholewindow, splitwindow, adaptivewindow)
# const AVAIL_TREATMENTS = (:aggregate, :reducesize)

const WIN_PARAMS = Dict(
    movingwindow   => (window_size = 1024, window_step = 512),
    wholewindow    => NamedTuple(),
    splitwindow    => (nwindows = 20),
    adaptivewindow => (nwindows = 20, relative_overlap = 0.5)
)

# """
# Abstract type for dataset configuration outputs
# """
# abstract type AbstractDatasetConfig end

# """
# Abstract type for dataset outputs
# """
# abstract type AbstractDataset end

# """
# Abstract type for dataset train, test and validation indexing
# """
# abstract type AbstractIndexCollection end

"""
Abstract type for feature struct
"""
abstract type AbstractFeature end

# """
#     DatasetInfo{F<:Base.Callable, R<:Real, I<:Integer, RNG<:AbstractRNG} <: AbstractDatasetConfig

# An immutable struct containing dataset configuration and metadata.
# It is included in ModelConfig and Dataset structs,
# In a ModelConfig object, it is reachable through the `ds.info` field. 

# # Fields
# - `algo::Symbol`:
#     Algorithm type, can be :classification, or :regression.
# - `treatment::Symbol`: 
#     Data treatment method, specify the behaviour of data reducing if dataset is composed of time-series.
#     :aggregate, time-series will be reduced to a scalar (propositional case).
#     :reducesize, time-series will be windowed to reduce size.
# - `features::Vector{F}`: 
#     Features functions applied to the dataset.
# - `train_ratio::R`: 
#     Ratio of training data (0-1), specify the ratio between train and test partitions,
#     the higher the ratio, the more data will be used for training.
# - `valid_ratio::R`: 
#     Ratio of validation data (0-1), spoecify the ratio between train and validation partitions,
#     the higher the ratio, the more data will be used for validation.
#     If `valid_ratio` is unspecified, no validation data will be used.
# - `shuffle::Bool`: 
#     Whether to shuffle data during train, validation and test partitioning.
# - `stratified::Bool`: 
#     Whether to use cross-validation stratified sampling technique.
# - `nfolds::I`: 
#     Number of cross-validation folds.
# - `rng::RNG`: 
#     Random number generator.
# - `winparams::Union{NamedTuple, Nothing}`: 
#     Window parameters: NamedTuple should have the following fields:
#     whole window (; type=wholewindow)
#     adaptive window (type=adaptivewindow, nwindows, relative_overlap),
#     moving window (type=movingwindow, nwindows, relative_overlap, window_size, window_step)
#     split window (type=splitwindow, nwindows).
# - `vnames::Union{Vector{Symbol}, Nothing}`: 
#     Variable names, usually dataset column names.
# """
# struct DatasetInfo{F<:Base.Callable, R<:Real, I<:Integer, RNG<:AbstractRNG} <: AbstractDatasetConfig
#     algo        :: Symbol
#     treatment   :: Symbol
#     features    :: Vector{F}
#     train_ratio :: R
#     valid_ratio :: R
#     shuffle     :: Bool
#     stratified  :: Bool
#     nfolds      :: I
#     rng         :: RNG
#     winparams   :: Union{NamedTuple, Nothing}
#     vnames      :: Union{Vector{Symbol}, Nothing}
# end

# function DatasetInfo(
#     algo::Symbol,
#     treatment::Symbol,
#     features::AbstractVector{F},
#     train_ratio::R,
#     valid_ratio::R,
#     shuffle::Bool,
#     stratified::Bool,
#     nfolds::I,
#     rng::RNG,
#     winparams::Union{NamedTuple, Nothing},
#     vnames::Union{AbstractVector{<:Union{AbstractString,Symbol}}, Nothing}
# ) where {F<:Base.Callable, R<:Real, I<:Integer, RNG<:AbstractRNG}
#     # Validate ratios
#     0 ≤ train_ratio ≤ 1 || throw(ArgumentError("train_ratio must be between 0 and 1"))
#     0 ≤ valid_ratio ≤ 1 || throw(ArgumentError("valid_ratio must be between 0 and 1"))

#     converted_vnames = isnothing(vnames) ? nothing : Vector{Symbol}(Symbol.(vnames))

#     DatasetInfo{F,R,I,RNG}(
#         algo, treatment, features, train_ratio, valid_ratio,
#         shuffle, stratified, nfolds, rng, winparams, converted_vnames
#     )
# end

# function Base.show(io::IO, info::DatasetInfo)
#     println(io, "DatasetInfo:")
#     for field in fieldnames(DatasetInfo)
#         value = getfield(info, field)
#         println(io, "  ", rpad(String(field) * ":", 15), value)
#     end
# end

# """
#     TT_indexes{T<:Integer} <: AbstractVector{T}

# A struct that stores indices for train-validation-test splits of a dataset,
# used in Dataset struct.

# # Fields
# - `train::Vector{T}`: Vector of indices for the training set
# - `valid::Vector{T}`: Vector of indices for the validation set
# - `test::Vector{T}`:  Vector of indices for the test set
# """
# struct TT_indexes{T<:Integer} <: AbstractIndexCollection
#     train       :: Vector{T}
#     valid       :: Vector{T}
#     test        :: Vector{T}
# end

# function TT_indexes(
#     train::AbstractVector{T},
#     valid::AbstractVector{T},
#     test::AbstractVector{T}
# ) where {T<:Integer}
#     TT_indexes{T}(train, valid, test)
# end

# Base.show(io::IO, t::TT_indexes) = print(io, "TT_indexes(train=", t.train, ", validation=", t.valid, ", test=", t.test, ")")
# Base.length(t::TT_indexes) = length(t.train) + length(t.valid) + length(t.test)

# function _create_views(X, y, tt, stratified::Bool)
#     if stratified
#         Xtrain = view.(Ref(X), getfield.(tt, :train), Ref(:))
#         Xvalid = view.(Ref(X), getfield.(tt, :valid), Ref(:))
#         Xtest  = view.(Ref(X), getfield.(tt, :test), Ref(:))
#         ytrain = view.(Ref(y), getfield.(tt, :train))
#         yvalid = view.(Ref(y), getfield.(tt, :valid))
#         ytest  = view.(Ref(y), getfield.(tt, :test))
#     else
#         Xtrain = @views X[tt.train, :]
#         Xvalid = @views X[tt.valid, :]
#         Xtest  = @views X[tt.test, :]
#         ytrain = @views y[tt.train]
#         yvalid = @views y[tt.valid]
#         ytest  = @views y[tt.test]
#     end
#     return Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest
# end

# """
#     Dataset{T<:AbstractDataFrame,S} <: AbstractDataset

# An immutable struct that efficiently stores dataset splits for machine learning.

# # Fields
# - `X::T`: The feature matrix as a DataFrame
# - `y::S`: The target vector
# - `tt::Union{TT_indexes{I}, Vector{TT_indexes{I}}}`: Train-test split indices
# - `info::DatasetInfo`: Dataset metadata and configuration
# - `Xtrain`, `Xvalid`, `Xtest`: Data views for features
# - `ytrain`, `yvalid`, `ytest`: Data views for targets
# """
# struct Dataset{T<:AbstractDataFrame,S} <: AbstractDataset
#     X           :: T
#     y           :: S
#     tt          :: Union{TT_indexes, AbstractVector{<:TT_indexes}}
#     info        :: DatasetInfo
#     Xtrain      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
#     Xvalid      :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
#     Xtest       :: Union{SubDataFrame{T}, Vector{<:SubDataFrame{T}}}
#     ytrain      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
#     yvalid      :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}
#     ytest       :: Union{SubArray{<:eltype(S)}, Vector{<:SubArray{<:eltype(S)}}}

#     function Dataset(X::T, y::S, tt, info) where {T<:AbstractDataFrame,S}
#         if info.stratified
#             Xtrain = view.(Ref(X), getfield.(tt, :train), Ref(:))
#             Xvalid = view.(Ref(X), getfield.(tt, :valid), Ref(:))
#             Xtest  = view.(Ref(X), getfield.(tt, :test), Ref(:))
#             ytrain = view.(Ref(y), getfield.(tt, :train))
#             yvalid = view.(Ref(y), getfield.(tt, :valid))
#             ytest  = view.(Ref(y), getfield.(tt, :test))
#         else
#             Xtrain = @views X[tt.train, :]
#             Xvalid = @views X[tt.valid, :]
#             Xtest  = @views X[tt.test, :]
#             ytrain = @views y[tt.train]
#             yvalid = @views y[tt.valid]
#             ytest  = @views y[tt.test]
#         end

#         new{T,S}(X, y, tt, info, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest)
#     end
# end

# function Base.show(io::IO, ds::Dataset)
#     println(io, "Dataset:")
#     println(io, "  X shape:        ", size(ds.X))
#     println(io, "  y length:       ", length(ds.y))
#     if ds.tt isa AbstractVector
#         println(io, "  Train/Valid/Test:     ", length(ds.tt), " folds")
#     else
#         println(io, "  Train indices:  ", length(ds.tt.train))
#         println(io, "  Valid indices:  ", length(ds.tt.valid))
#         println(io, "  Test indices:   ", length(ds.tt.test))
#     end
#     print(io, ds.info)
# end

"""
    Feature{V<:Number, T<:Union{Symbol, String}} <: AbstractFeature

A parametric struct that represents a feature extracted from time series data.

# Type Parameters
- `V`: Type of the feature value (must be a subtype of `Number`)
- `T`: Type of the variable name (must be either `Symbol` or `String`)

# Fields
- `value::V`: The numerical value of the feature
- `var::T`: The variable name/identifier
- `feats::Symbol`: The feature extraction function name
- `nwin::Int`: The window number (must be positive)

# Constructors
```julia
Feature(value::Number, var::Union{Symbol,String}, feats::Symbol, nwin::Integer)
"""
struct Feature{V<:Number, T<:Union{Symbol, String}} <: AbstractFeature
    value :: V
    var   :: T
    feats :: Symbol
    nwin  :: Int

    function Feature(value::Number, var::Union{Symbol,String}, feats::Symbol, nwin::Integer)
        nwin > 0 || throw(ArgumentError("Window number must be positive"))
        new{typeof(value), typeof(var)}(value, var, feats, nwin)
    end
end

# Pretty printing
Base.show(io::IO, f::Feature) = print(io, 
    "Feature($(f.value), $(f.var), $(f.feats), window=$(f.nwin))")

# Value access methods
Base.getproperty(f::Feature, s::Symbol) = getfield(f, s)
Base.propertynames(::Feature) = (:value, :var, :feats, :nwin)

# Conversion methods for NaN handling
Base.convert(::Type{Feature}, x::Missing) = Feature(NaN, :missing, :none, 1)
Base.convert(::Type{Feature}, x::Nothing) = Feature(NaN, :nothing, :none, 1)

# Test if value is NaN
Base.isnan(f::Feature) = isnan(f.value)

# Numeric comparisons
Base.isless(f::Feature, x::Number) = isless(f.value, x)
Base.isless(x::Number, f::Feature) = isless(x, f.value)
Base.isless(f1::Feature, f2::Feature) = isless(f1.value, f2.value)

# Convert to number for arithmetic operations
Base.convert(::Type{Number}, f::Feature) = f.value
Base.convert(::Type{Float64}, f::Feature) = convert(Float64, f.value)

# Forward numeric operations to the value field
for op in (:+, :-, :*, :/, :^)
    @eval Base.$op(f::Feature, x::Number) = $op(f.value, x)
    @eval Base.$op(x::Number, f::Feature) = $op(x, f.value)
end

# Get variable name
variable_name(f::Feature) = f.var
# Get feature type
feature_type(f::Feature) = f.feats
# Get window number
window_number(f::Feature) = f.nwin
