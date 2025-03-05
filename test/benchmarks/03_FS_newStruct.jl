using DataFrames
using SoleData
using SoleFeatures
using DataStructures
using HypothesisTests
using Random
using Catch22
using StatsBase
using SimpleCaching
using Dates
using ConfigEnv
using Distributions
using CSV
using Statistics
using MLBase
using NaturalSort
using CategoricalArrays

# include("/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/src/filters/mutual_info.jl")

# struct PyMutualInformationClassif{T <: SoleFeatures.AbstractLimiter} <: SoleFeatures.AbstractMutualInformationClassif{T}
#     limiter::T
# end

# SoleFeatures.is_supervised(::PyMutualInformationClassif) = true
# SoleFeatures.is_unsupervised(::PyMutualInformationClassif) = false

# function SoleFeatures.score(
#     X::AbstractDataFrame,
#     y::AbstractVector{<:Integer},
#     selector::PyMutualInformationClassif{<:SoleFeatures.AbstractLimiter}
# )::Vector{Float64}
#     scores = mutual_info_classif(X, y)
#     return Float64.(scores)
# end

# # Ranking
# PyMutualInformationClassifRanking(nbest) = PyMutualInformationClassif(RankingLimiter(nbest, true))

Base.nameof(f::SuperFeature) = getname(f) # wrap for Catch22

macro safeconst(ex)
	if ex.head != :(=)
		throw(ArgumentError("@safeconst need to be used before an assignment."))
	end

	if !isa(ex.args[1], Symbol)
		throw(ArgumentError("only single assignment are allowed in @safeconst."))
	end

	sym = ex.args[1]
	ex = Expr(:const, ex)

	return quote
		if !(@isdefined $(sym))
			$(esc(ex))
		end
    end
end

# Fixed number moving windows
abstract type AbstractMovingWindows end
abstract type AbstractMovingWindowsIndex end
include("/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/src/experimental/windows/windows.jl")
include("/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/src/experimental/extraction.jl")
include("/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/src/experimental/windows/data-filters.jl")

# ---------------------------------------------------------------------------- #
#                                    types                                     #
# ---------------------------------------------------------------------------- #
const ABT = Union{NamedTuple{(:aggrby,:aggregatef,:group_before_score)}, Nothing}

# /------------------- FUNCTIONS -----------------------------

# TODO: super ugly!!! this should be done by SoleFeatures but better than this
# SoleFeatures.is_unsupervised(::Any) = false
# is_unsupervised = SoleFeatures.is_unsupervised

# """
#     _fs(X, [y,] selector, limiter)

# Perform a feature selection using `selector` limiting the variables selected
# using `limiter`. `X` is the dataset as `AbstractDataFrame`.

# If a supervised selector is passed the `y` parameter is needed: an `AbstractVector`
# of labels.
# """
# function _fs(
#     X::AbstractMatrix,
#     y::Union{AbstractVector,Nothing},
#     Xinfo::AbstractVector{<:SoleFeatures.InfoFeat},
#     selector::SoleFeatures.AbstractFeaturesSelector,
#     limiter::SoleFeatures.AbstractLimiter
# )::Tuple{Vector{Int},Vector{SoleFeatures.Score}}
#     scores = isnothing(y) || SoleFeatures.is_unsupervised(selector) ?
#         SoleFeatures.score(X, selector) :
#         SoleFeatures.score(X, y, selector)
#     # limit è il metodo per fare i tagli
#     idxes = SoleFeatures.limit(scores, limiter)

#     # Store scores directly in the Xinfo objects
#     # for (i, score) in enumerate(scores)
#     #     setfield!(Xinfo[i], score_target, score)
#     # end

#     # return score, idxes
#     return idxes, [SoleFeatures.Score(i.id, score) for (i, score) in zip(Xinfo, scores)]
# end

# function _fs(
#     X::AbstractMatrix,
#     Xinfo::AbstractVector{<:SoleFeatures.InfoFeat},
#     selector::SoleFeatures.AbstractFeaturesSelector,
#     limiter::SoleFeatures.AbstractLimiter
# )::Tuple{Vector{Int},Vector{SoleFeatures.Score}}
#     return _fs(X, nothing, Xinfo, selector, limiter)
# end

"""
    group_names(X, aggrby; groups_separator = "@@@")

Retrieve unique group names of `X` by splitting column names using
`groups_separator` and looking only at `aggrby` portion of the name.
"""
function group_names(
    Xinfo::AbstractVector{<:SoleFeatures.InfoFeat},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{<:AbstractString}}

    # Verify all properties exist in the InfoFeat objects
    for prop in aggrby
        if !hasproperty(first(Xinfo), prop)
            throw(ArgumentError("InfoFeat objects don't have property: $prop"))
        end
    end

    # get unique group names
    # ixs = sort([aggrby...])

    # return unique([getfield(x, ag) for x in Xinfo for ag in [aggrby...]])

        # Get unique combinations of values for the specified fields
        unique_combinations = unique([
            Tuple(string(getfield(info, field)) for field in aggrby)
            for info in Xinfo
        ])
        
        # Convert tuples to vectors
        return [collect(combination) for combination in unique_combinations]
end

# function group_names(X::AbstractDataFrame, args...; kwargs...)
#     group_names(names(X), args...; kwargs...)
# end

"""
    _is_part_of_the_group(group_name, column_name, ixs)

Return whether `column_name` is part of the group identified by
`group_name` if looking only at `ixs` part of `column_name`.

Note that `column_name` has to be a column name already splitted. If the name
is passed as a string it is used a `groups_separator` to split it (default: "@@@").
"""
function _is_part_of_the_group(
    group_name::AbstractVector{<:AbstractString},
    column_name::AbstractVector{<:AbstractString},
    ixs::AbstractVector{<:Integer}
)::Bool
    return group_name == column_name[ixs]
end
function _is_part_of_the_group(
    group_name::AbstractVector{<:AbstractString},
    column_name::AbstractString,
    ixs::AbstractVector{<:Integer};
    groups_separator::AbstractString = _SEPARATOR
)::Bool
    return _is_part_of_the_group(
        group_name,
        split(column_name, groups_separator),
        ixs
    )
end

# function group_id_by_aggrby(
#     Xinfo::AbstractVector{<:SoleFeatures.InfoFeat},
#     aggrby::Tuple{Vararg{Symbol}}
# )::Vector{Vector{Int}}
#     g_names = group_names(Xinfo, aggrby)

#     # ixs = sort([aggrby...])
#     # res = [findall(Xname -> _is_part_of_the_group(cur_g_name, Xname, ixs; groups_separator = groups_separator), Xnames)
#     #         for cur_g_name in g_names]

#     # Get all unique combinations of values for the specified fields
#     value_combinations = unique([
#         Tuple(getfield(info, field) for field in aggrby)
#         for info in Xinfo
#     ])
#     # For each unique combination, find all indices with matching values
#     ixs = [
#         findall(i -> Tuple(getfield(Xinfo[i], field) for field in aggrby) == combination, 
#                 1:length(Xinfo))
#         for combination in value_combinations
#     ]

#     # @assert !any(isempty.(res)) "Some of the groups are empty!"
#     any(isempty.(ixs)) && throw(ErrorException("Some of the groups are empty!"))
#     # return res
#     return ixs
# end

# Xinfo = "Y[Hand tip r]", :minimum, 3), "Y[Hand tip r]", :minimum, 4), SoleFeatures.InfoFeat{String}(2, "Y[Hand tip l]", :maximum, 2), SoleFeatures.InfoFeat{String}(2, "Y[Hand tip l]", :maximum, 3), SoleFeatures.InfoFeat{String}(2, "Y[Hand tip l]", :maximum, 4), SoleFeatures.InfoFeat{String}(2, "Y[Hand tip l]", :maximum, 5), SoleFeatures.InfoFeat{String}(5, "Y[Hand tip r]", :maximum, 2), SoleFeatures.InfoFeat{String}(20, "Y[Thumb l]", :maximum, 3), SoleFeatures.InfoFeat{String}(20, "Y[Thumb l]", :maximum, 4), SoleFeatures.InfoFeat{String}(2, "Y[Hand tip l]", :mean, 3), SoleFeatures.InfoFeat{String}(2, "Y[Hand tip l]", :mean, 4)]
# Xnames = ["Z[Hand tip l]@@@W2(6,0.05)@@@minimum", "X[Hand tip r]@@@W2(6,0.05)@@@minimum", "Y[Elbow l]@@@W1(6,0.05)@@@maximum", "Z[Elbow l]@@@W1(6,0.05)@@@maximum", "X[Elbow r]@@@W1(6,0.05)@@@maximum", "Y[Elbow r]@@@W1(6,0.05)@@@maximum", "Y[Hand tip l]@@@W2(6,0.05)@@@maximum", "Z[Thumb l]@@@W5(6,0.05)@@@maximum", "X[Thumb r]@@@W5(6,0.05)@@@maximum", "Z[Elbow l]@@@W1(6,0.05)@@@mean", "X[Elbow r]@@@W1(6,0.05)@@@mean"]

# g_names = Vector{<:AbstractString}[["Y[Hand tip r]"], ["Y[Hand tip l]"], ["Y[Thumb l]"]]
# g_indices = [[1, 2, 7], [3, 4, 5, 6, 10, 11], [8, 9]]
# g_names = Vector{<:AbstractString}[SubString{String}["Z[Hand tip l]"], SubString{String}["X[Hand tip r]"], SubString{String}["Y[Elbow l]"], SubString{String}["Z[Elbow l]"], SubString{String}["X[Elbow r]"], SubString{String}["Y[Elbow r]"], SubString{String}["Y[Hand tip l]"], SubString{String}["Z[Thumb l]"], SubString{String}["X[Thumb r]"]]
# length(g_names) = 9

# function group_indices_by_column_names(X::AbstractDataFrame, args...; kwargs...)
#     return group_indices_by_column_names(names(X), args...; kwargs...)
# end

function group_by_column_names(
    X::AbstractDataFrame,
    aggrby::Tuple{Vararg{Integer}};
    groups_separator::AbstractString = _SEPARATOR
)::Vector{SubDataFrame}
    return [(@view X[:,idxs]) for idxs in group_indices_by_column_names(X, aggrby; groups_separator = groups_separator)]
end

# """
# Perform feature selection on groups

# ## PARAMS

# - `X`: the dataset in the form of `AbstractDataFrame`;
# - `y`: the labels if the dataset is supervised; if the passed `selector` is supervised it has to be different from `nothing`;
# - `selector`: the feature selection algorithm of type `AbstractFeaturesSelector`;
# - `limiter`: the policy used to select the features to of type `AbstractLimiter`;
# - `aggrby`: it is a tuple describing the portion of the column name to use to determine groups;
# - `groups_separator`: the substring used to split the `DataFrame` names; default value is `"@@@"`.
# - `aggregatef`: use this function to aggregate results from groups; default value is `identity`;
# - `group_before_score`: it the passed `AbstractFeaturesSelector` is multivariate it can lead to
#     different results to calculate scores after or before grouping variables by `aggrby` parameter.

# ## RETURN

# - sel_idxes::Vector{Int},
# - group_scores::Vector Aggregated score for each group, if aggregatef is identity function than group_scores will be equal to scores
# - scores:::Vector Not aggregated scores for each group

# # TODO: expand documentation
# """
# function _fsgroup(
#     X::AbstractMatrix,
#     y::Union{AbstractVector,Nothing},
#     Xinfo::AbstractVector{<:SoleFeatures.InfoFeat},
#     selector::SoleFeatures.AbstractFeaturesSelector,
#     limiter::SoleFeatures.AbstractLimiter,
#     aggrby::Tuple{Vararg{Symbol}};
#     aggregatef::Function = mean,
#     group_before_score::Union{Val{true},Val{false}} = Val(true),
# # )::Tuple{Vector{Int},Vector{Vector{Int}},Vector{<:Real},Vector{Vector{<:Real}}}
# )::Tuple{Vector{Int},Vector{Vector{Int}},Vector{SoleFeatures.GroupScore},Vector{Vector{<:Real}}}
#     g_indices = group_id_by_aggrby(Xinfo, aggrby)

#     scores = []
#     groups_score = SoleFeatures.GroupScore[]
#     # groups_score = Vector(undef, length(g_indices))

#     if group_before_score isa Val{true}
#         # === group and then evaluate score internally to each group ===
#         for (i, cur_g_indices) in enumerate(g_indices)

#             s = isnothing(y) || SoleFeatures.is_unsupervised(selector) ?
#                 SoleFeatures.score(X[:,cur_g_indices], selector) :
#                 SoleFeatures.score(X[:,cur_g_indices], y, selector)

#             push!(scores, s) # save scores of variables of current group
#             grp = Tuple(Symbol.(collect(getfield(first(Xinfo[cur_g_indices]), a) for a in aggrby)))
#             push!(groups_score, SoleFeatures.GroupScore(grp, aggregatef(s))) # save scores of variables of current group
#             # groups_score[i] = aggregatef(s) # save aggregated group score
#         end
#     else
#         # === calculate scores for all variables and then group ===
#         allscores = isnothing(y) || SoleFeatures.is_unsupervised(selector) ?
#             SoleFeatures.score(X, selector) :
#             SoleFeatures.score(X, y, selector)

#         for (i, cur_g_indices) in enumerate(g_indices)
#             push!(scores, allscores[cur_g_indices]) # save scores of variables of current group
#             grp = Tuple(Symbol.(collect(getfield(first(Xinfo[cur_g_indices]), a) for a in aggrby)))
#             push!(groups_score, SoleFeatures.GroupScore(grp, aggregatef(allscores[cur_g_indices])))
#             # groups_score[i] = aggregatef(allscores[cur_g_indices]) # save aggregated group score
#         end
#     end

#     # convert groups_score from Vector{Any} type to Vector{Type of first element} type
#     groups_score = convert.(typeof(groups_score[1]), groups_score)

#     # # apply limiter on groups
#     sel_idxes = SoleFeatures.limit(groups_score, limiter)

#     # first element: index of the selected groups
#     # second element: indices of variables for each group
#     # third element: score of each group
#     # fourth element: score of each variable grouped
#     # return sel_idxes, g_indices, groups_score, scores
#     return sel_idxes, g_indices, groups_score, scores
# end

# function _fsgroup(
#     X::AbstractMatrix,
#     Xinfo::AbstractVector{<:SoleFeatures.InfoFeat},
#     selector::SoleFeatures.AbstractFeaturesSelector,
#     limiter::SoleFeatures.AbstractLimiter,
#     aggrby::Tuple{Vararg{Symbol}};
#     kwargs...
# # )::Tuple{Vector{Int},Vector{Vector{Int}},Vector{<:Real},Vector{Vector{<:Real}}}
# )::Tuple{Vector{Int},Vector{Vector{Int}},Vector{SoleFeatures.GroupScore},Vector{Vector{<:Real}}}
#     return _fsgroup(X, nothing, Xinfo, selector, limiter, aggrby; kwargs...)
# end

"""
    validate_features(X, y = nothing; method = :pvalue)

This function performs validation of variables (it usually is useful after
a feature selection step, see [`feature_selection`](@ref)).

## PARAMETERS

- `X`: is the dataset in the form of an `AbstractDataFrame`;
- `y`: are the labels of the dataset (`AbstractVector`), it is optional
depending on the selected `method`;
- `method`: it the method to use to perform the validation;
- `aggrf`: when a more than binary class problem is inputed, results
need to be aggregated; this is done using this function (default: `minimum`).

## AVAILABLE METHODS

- `:pvalue`: this method performs the `MannWhitneyUTest` on all classes
to determine whether the feature is distributed significantly differently
between the classes (one vs all). This method is supervised and needs
`y` not to be `nothing`.

Currently `:pvalue` is the only implemented method.
"""
function validate_features(
    X::AbstractDataFrame,
    y::Union{AbstractVector,Nothing} = nothing;
    method::Symbol = :pvalue,
    aggrf::Function = minimum,
)::Tuple{Any,Vector{Int}}

    if method == :pvalue
        if isnothing(y)
            throw(ArgumentError("Selected method :pvalue need the dataset " *
                "to be supervised to work; passed `y` = $y"))
        end

        score, indices = _fs(
            X, y,
            SoleFeatures.StatisticalMajority(MannWhitneyUTest; versus = :ova),
            SoleFeatures.StatisticalLimiter(
                SoleFeatures.AtLeastLimiter(
                    SoleFeatures.ThresholdLimiter(0.05, <=),
                    1
                )
            )
        )

        # TODO: consider keeping only the `else` code
        if length(unique(y)) == 2
            # return minimum pvalue score (for all features) and indices of (only) validated features
            return aggrf.(eachrow(score[:,1])), indices
        else
            # return
            return aggrf.(eachrow(score)), indices
        end
    else
        throw(ArgumentError("Unsupported method :$(method)! Available are: :pvalue"))
    end
end

"""
TODO: documentation
"""
function _fix_nan_inf_dataset!(
    X::AbstractDataFrame,
    y::Union{Nothing,AbstractVector} = nothing;
    replace_special_float::Bool = true,
    convert_nan_to = 0,
    convert_inf_to = 0,
    convert_ninf_to = 0,
    nan_danger_fraction_threshold::AbstractFloat = 0.05,
    remove_too_nan_instance::Bool = false,
)
    if (replace_special_float)
        replace!.(eachcol(X), [NaN => convert_nan_to])
        replace!.(eachcol(X), [Inf => convert_inf_to])
        replace!.(eachcol(X), [-Inf => convert_ninf_to])
    end

    if remove_too_nan_instance
        nanrowidxes = findall(r -> any(isnan, r), eachrow(X));
        if (length(nanrowidxes) > 0)
            @warn "Passed dataset contains $(length(nanrowidxes)) instances " *
                "containing NaN values.\nInstances containing NaN values " *
                "will be removed"

            if (length(nanrowidxes) > Int(ceil(size(X, 1) * nan_danger_fraction_threshold)))
                @warn "DANGER: Passed dataset contains too many NaN instances"
            end

            deleteat!(X, nanrowidxes)
            !isnothing(y) && deleteat!(y, nanrowidxes)
        end
    end

    return X, y
end

# # TODO if working move to SoleFeatures
# function _features_groupby(
#     Xinfo::Vector{<:SoleFeatures.InfoFeat},
#     aggrby::Tuple{Vararg{Symbol}}
# )::Vector{Vector{Int}}
#     g_names = group_names(Xinfo, aggrby)

#     res = Dict{Any, Vector{Int}}()
#     for (i, g) in enumerate(Xinfo)
#         key = Tuple(getproperty(g, field) for field in group)
#         push!(get!(res, key, Int[]), i)
#     end
#     return collect(values(res))  # Return the grouped indices
# end

"""
    _normalize_dataset!(
        X::AbstractMatrix{T},
        Xinfo::Vector{<:SoleFeatures.InfoFeat};
        min_quantile::AbstractFloat=0.00,
        max_quantile::AbstractFloat=1.00,
        group::Tuple{Vararg{Symbol}}=(:nwin, :feat),
    ) where {T<:Number}

Normalize the dataset matrix `X` by applying min-max normalization to groups of features.

## Parameters
- `X`: The input matrix to be normalized in-place
- `Xinfo`: A vector of feature information objects that contain metadata about each feature
- `min_quantile`: The quantile to use as the minimum value (default: 0.00)
  - When set to 0.00, uses the absolute minimum value
  - Higher values (e.g., 0.05) ignore lower outliers by using the specified quantile instead
- `max_quantile`: The quantile to use as the maximum value (default: 1.00)
  - When set to 1.00, uses the absolute maximum value
  - Lower values (e.g., 0.95) ignore upper outliers by using the specified quantile instead
- `group`: A tuple of symbols representing fields in the `InfoFeat` objects to group by (default: (:nwin, :feat))
  - Features with the same values for these fields will be normalized together
  - For example, with the default (:nwin, :feat), features from the same window and of the same type
    will be normalized as a group, preserving their relative scale

## Details
The function performs group-wise normalization, which is essential when working with features that 
should maintain their relative scales. For example, when working with time series data, different 
measures (min, max, mean) applied to the same window should be normalized together to preserve 
their relationships.
"""
function _normalize_dataset!(
    X::AbstractMatrix{T},
    Xinfo::Vector{<:SoleFeatures.InfoFeat};
    min_quantile::AbstractFloat=0.00,
    max_quantile::AbstractFloat=1.00,
    group::Tuple{Vararg{Symbol}}=(:nwin, :feat),
) where {T<:Number}
    for g in _features_groupby(Xinfo, group)
        SoleFeatures.minmax_normalize!(
            view(X, :, g);
            min_quantile = min_quantile,
            max_quantile = max_quantile,
            col_quantile = false
        )
    end
end

@safeconst FSMidResults =  NamedTuple{
    (:extraction_column_names,:fs_mid_results),
    Tuple{Vector{String},Vector{NamedTuple{
        (:score,:indices,:name2score,:group_aggr_func,:group_indices,:aggrby)
    }}}
}

"""
TODO: documentation

# Feature Selection with Aggregation Control

## Overview
The `feature_selection` function allows precise control over how feature aggregation
is applied during the multi-step feature selection process.

## Aggregation Parameter (`aggrby`)
The `aggrby` parameter can be provided in two ways:

1. **Single NamedTuple**: When provided as a single NamedTuple (not a vector), 
   aggregation is only applied during the final step of feature selection.
   The function automatically creates a vector where:
   - All positions except the last contain `nothing`
   - The last position contains the provided aggregation parameters

2. **Vector of NamedTuples**: When provided as a vector, each element specifies 
   the aggregation behavior for the corresponding step in `fs_methods`.
"""
function feature_selection(
    X::AbstractMatrix{T},
    y::Union{Nothing,AbstractVector},
    Xinfo::Vector{<:SoleFeatures.InfoFeat};

    # groups_separator::AbstractString = _SEPARATOR,

    # ex_windows::AbstractVector = [ FixedNumMovingWindows(5, 0.05)... ],
    # ex_measures::AbstractVector{Union{Function, SuperFeature}} = [minimum, maximum, mean],

    # cosa vuoi fare al dataset, crea la tripla var, win, feats
    # extract_tuples::AbstractVector = vec(collect(Iterators.product(names(X), ex_windows, ex_measures))),

    # tipo di aggregazione che si vuole alla fine
    aggrby::Union{ABT,AbstractVector{<:ABT}} = (
        aggrby = (:var,),
        aggregatef = length, # NOTE: or mean, minimum, maximum to aggregate scores instead of just counting number of selected features for each group
        group_before_score = true,
    ),

    fs_methods::AbstractVector{<:NamedTuple{(:selector, :limiter)}} = [
        ( # STEP 1: unsupervised variance-based filter
            selector = SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.5),
        ),
        ( # STEP 2: supervised Mutual Information filter
            selector = SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.1),
        ),
        ( # STEP 3: group results by variable
            selector = IdentityFilter(),
            limiter = SoleFeatures.IdentityLimiter(),
        ),
    ],

    # fix_special_floats::Bool = false,
    # fix_special_floats_kwargs::NamedTuple = NamedTuple(),
    norm::Bool = false,
    normalize_kwargs::NamedTuple = NamedTuple(),

    cache_extracted_dataset::Union{Nothing,AbstractString} = nothing,
    return_mid_results::Union{Val{true},Val{false}} = Val(true),
# )::Union{DataFrame,Tuple{DataFrame,FSMidResults}} where {T<:Number}
) where {T<:Number}

    # ==================== PREPARE INPUTS ====================

    if !(aggrby isa AbstractVector)
        # when aggrby is not a Vector assume that the user want to perform aggregation
        #    only during the last step of feature selection TODO: document this properly!!!
        aggrby = push!(Union{Nothing,NamedTuple}[fill(nothing, max(length(fs_methods)-1, 0))...], aggrby)
    end

    # ==================== PREPARE LABELS ====================

    y_coded = @. CategoricalArrays.levelcode(y)

    # ================== DATASET EXTRACTION ==================

    # QUI inizia feature selection
    # extract new dataset
    # newX = begin
    #     local ced
    #     local _extr
    #     ced = cache_extracted_dataset
    #     _extr = extract
    #     # TODO: this Float64 is a strong assumption!
    #     Float64.(@scache_if !isnothing(ced) "dse" ced _extr(X, extract_tuples))
    # end

    # # groups_separator = "@@@"
    # if groups_separator != _SEPARATOR
    #     rename!(x -> replace(x, _SEPARATOR => groups_separator), newX)
    # end
    # extraction_column_names = names(newX)


    # =================== SPECIAL FLOAT FIX ===================

    # if fix_special_floats
    #     @warn "DANGER!!! It is really discouraged to call this function " *
    #         "`fix_special_floats` set to `true`"
    #     fix_special_floats_kwargs = merge(fix_special_floats_kwargs, (remove_too_nan_instance = false,))
    #     _fix_nan_inf_dataset!(newX, y; fix_special_floats_kwargs...)
    #     # FIXME: this function could alter the length o `y` and create
    #     #          heavy inconsistencies!!! (this is why I forced
    #     #          `remove_too_nan_instance` to false)
    # end

    # ================== DATASET NORMALIZATION ==================

    norm && _normalize_dataset!(X, Xinfo; normalize_kwargs...)

    # =================== NO FEATURE SELECTION ==================

    # # if no feature selector was passed we can assume the user just wanted to extract features from dataset
    # if length(fs_methods) == 0
    #     if isa(return_mid_results, Val{true})
    #         return X, NamedTuple()
    #     else
    #         return X
    #     end
    # end

    # ===================== FEATURE SELECTION ===================

    # questo serve solo per generare grafici
    # fs_mid_results = NamedTuple{(:score,:indices,:name2score,:group_aggr_func,:group_indices,:aggrby)}[]
    fs_mid_results = NamedTuple{(:score, :indices,:group_aggr_func,:group_indices,:aggrby)}[]

    for (fsm, gfs_params) in zip(fs_methods, aggrby)
        current_dataset_col_slice = 1:size(X, 2)

         # pick survived columns only
        for i in 1:length(fs_mid_results)
            current_dataset_col_slice = current_dataset_col_slice[fs_mid_results[i].indices]
        end

        currX = X[:,current_dataset_col_slice]
        currXinfo = Xinfo[current_dataset_col_slice]

        dataset_param = isnothing(y_coded) || SoleFeatures.is_unsupervised(fsm.selector) ? 
            (currX, currXinfo) : 
            (currX, y_coded, currXinfo)

        idxes, score, g_indices =
            if isnothing(gfs_params)
                # perform normal feature selection
                SoleFeatures._fs(dataset_param..., fsm...)..., nothing
            else
                # perform aggregated feature selection
                sel_g_indices, g_scores, g_indices, grouped_variable_scores = SoleFeatures._fsgroup(
                    dataset_param..., fsm..., gfs_params.aggrby;
                    aggregatef = gfs_params.aggregatef,
                    group_before_score = gfs_params.group_before_score
                )

                # find indices to re-sort the scores of all variables to their
                # original position in dataset columns
                old_sort = sortperm(vcat(g_indices...))

                vcat(g_indices[sel_g_indices]...), vcat(vcat(grouped_variable_scores...)[old_sort]...), g_indices
            end

        sort!(idxes)

        push!(fs_mid_results, (
            score = score,
            indices = idxes,
            group_aggr_func = isnothing(gfs_params) ? nothing : gfs_params.aggregatef,
            group_indices = g_indices,
            aggrby = isnothing(gfs_params) ? nothing : gfs_params.aggrby
        ))
    end

    dataset_col_slice = 1:size(X, 2)
    for i in 1:length(fs_mid_results)
        dataset_col_slice = dataset_col_slice[fs_mid_results[i].indices]
    end

    if isa(return_mid_results, Val{true})
        return X, X[:,dataset_col_slice], (extraction_column_names = Xinfo[dataset_col_slice], fs_mid_results = fs_mid_results)
    else
        return X[:,dataset_col_slice]
    end
end
feature_selection(X::AbstractDataFrame, args...; kwargs...) = feature_selection(Matrix(X), args...; kwargs...)

"""
TODO: docs
"""
function get_feature_names_at_step(fs::FSMidResults, i::Integer)
    if i < 1 || i > length(fs.fs_mid_results)
        throw(ArgumentError("passed `i` = $i but `fs` had $(length(fs.fs_mid_results)) steps"))
    end

    dataset_col_slice = 1:length(fsextraction_column_names)
    for j in 1:i
        dataset_col_slice = dataset_col_slice[fs.fs_mid_results[j].indices]
    end
end

"""
    study_feature_selection_mid_results(fs_mid_results)

Return a [`countmap`](@ref) for each "split name" of the features
extracted during the feature selection process and for each step where
"split name" is normally a split between "variable name", "window" and
"applied measure" (in this order) but "variable name" can be further
splitted if the dataset supported it.
"""
function study_feature_selection_mid_results(
    fs::FSMidResults;
    groups_separator::AbstractString = _SEPARATOR,
)::Vector{Vector{Dict{String,Int}}}
    extraction_column_names = fs.extraction_column_names
    fs_mid_results = fs.fs_mid_results

    splitted_names = split.(extraction_column_names, groups_separator)
    n_portions = length(first(splitted_names))

    # at step zero look at all columns
    dataset_col_slice = 1:length(extraction_column_names)

    res = Vector{Vector{Dict{String,Int}}}(undef, length(fs_mid_results)+1)
    for i in 0:length(fs_mid_results)
        # NOTE: iteration 0 will produce the "original dataset" study
        survived_splitted_names = split.(extraction_column_names[dataset_col_slice], groups_separator)

        res[i+1] = Vector{Dict{String,Int}}(undef, n_portions)
        for portion in 1:n_portions
            # count how many times `portion` occurred in survived columns
            res[i+1][portion] = countmap(getindex.(survived_splitted_names, portion))

            # add name portion that were never selected in i-th step
            for n in unique(getindex.(splitted_names, portion))
                if !haskey(res[i+1][portion], n)
                    res[i+1][portion][n] = 0
                end
            end
        end

        # update survived column indices
        if i < length(fs_mid_results)
            dataset_col_slice = dataset_col_slice[fs_mid_results[i+1].indices]
        end
    end

    return res
end

function pretty_print_feature_selection_study(
    res::AbstractVector{<:AbstractVector{<:AbstractDict{<:AbstractString,<:Integer}}};
    sort_keys::Bool = true,
    show_with_zero_occurs::Bool = true,
    show_initial_dataset::Bool = false
)::Nothing
    println("● Feature selection study result:")
    for fs_step in (show_initial_dataset ? 1 : 2):length(res)
        print("  ", fs_step == length(res) ? "└─ " : "├─ ")
        p_fs_step = fs_step == length(res) ? " " : "│"
        if fs_step == 1
            println("Initial dataset:")
        else
            println("FS step ", fs_step-1, ":")
        end
        for col_name_portion in 1:length(res[fs_step])
            print("  $p_fs_step    ", col_name_portion == length(res[fs_step]) ? "└─ " : "├─ ")
            p_col_name_portion = col_name_portion == length(res[fs_step]) ? " " : "│"
            part_name =
                if col_name_portion == length(res[fs_step])
                    "Measures"
                elseif col_name_portion == length(res[fs_step])-1
                    "Windows"
                else
                    string("Variable (pt: ", col_name_portion, "/", length(res[fs_step])-2, ")")
                end
            println(part_name, ":")
            nk = length(res[fs_step][col_name_portion])
            f = sort_keys ? v -> sort(v; lt = natural) : identity
            for (i, k) in enumerate(f(collect(keys(res[fs_step][col_name_portion]))))
                v = res[fs_step][col_name_portion][k]
                if show_with_zero_occurs || v != 0
                    print("  $p_fs_step    $p_col_name_portion    ", i == nk ? "└─ " : "├─ ")
                    println(k, " : ", v)
                end
            end
        end
    end
end

"""
    apply_feature_selection_mid_results_to_original_dataset(X, fs_mid_results)

This can be used only if the feature selection pipeline has only one "grouped"
feature selection step and it is the last one.

The parameter `fs_mid_results` is the second element of the Tuple returned by the function
[`feature_selection`](@ref) and `X` is the original dataset passed to it.
"""
function apply_feature_selection_mid_results_to_original_dataset(
    X::AbstractDataFrame,
    fs::FSMidResults;
    groups_separator::AbstractString = _SEPARATOR,
)::AbstractDataFrame
    fs_mid_results = fs.fs_mid_results
    extraction_column_names = fs.extraction_column_names

    if length(fs_mid_results) == 0
        return X
    end

    n_pieces_features = length(split(first(extraction_column_names), groups_separator))
    level = fs_mid_results[end].aggrby

    if !(all(isnothing, getproperty.(fs_mid_results[1:(end-1)], :group_aggr_func)) &&
            !isnothing(fs_mid_results[end].group_aggr_func))
        throw(ArgumentError("This function can be used only if the feature selection pipeline has " *
                "only one \"grouped\" feature selection step and it is the last one."))
    end

    if !all(≤(length(extraction_column_names) - 2), level)
        throw(ArgumentError("Can't apply window and/or measure `level` to the dataset.\n" *
            "`level` = $level, the split function returned `n_pieces_features` = $(n_pieces_features)."))
    end

    dataset_col_slice = 1:length(extraction_column_names)
    for i in 1:length(fs_mid_results)
        dataset_col_slice = dataset_col_slice[fs_mid_results[i].indices]
    end

    # retrieve group names to apply it to the
    ixs = sort([level...])
    selected_groups = [comp_name[ixs] for comp_name in split.(extraction_column_names[dataset_col_slice], groups_separator)]

    to_keep = Int[]
    for sg in selected_groups
        found = findall(cn -> _is_part_of_the_group(
                sg, cn, ixs;
                groups_separator = groups_separator
            ), names(X))
        if !isnothing(found)
            append!(to_keep, found)
        end
    end

    return X[:,sort(unique(to_keep))]
end

function variable_selection(
    X::AbstractDataFrame,
    y::Union{Nothing,AbstractVector},
    aggregatef::Function = length,
    group_before_score::Bool = true;

    groups_separator::AbstractString = _SEPARATOR,

    ex_windows::AbstractVector = [ FixedNumMovingWindows(5, 0.05)... ],
    ex_measures::AbstractVector{Union{Function, SuperFeature}} = [minimum, maximum, mean],

    extract_tuples::AbstractVector = vec(collect(Iterators.product(names(X), ex_windows, ex_measures))),

    fs_methods::AbstractVector{<:NamedTuple{(:selector, :limiter)}} = [
        ( # STEP 1: unsupervised variance-based filter
            selector = SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.5),
        ),
        ( # STEP 2: supervised Mutual Information filter
            selector = SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.1),
        ),
    ],

    fix_special_floats::Bool = false,
    fix_special_floats_kwargs::NamedTuple = NamedTuple(),
    normalize::Bool = false,
    normalize_kwargs::NamedTuple = NamedTuple(),

    cache_extracted_dataset::Union{Nothing,AbstractString} = nothing,
    return_mid_results::Union{Val{true},Val{false}} = Val(true),
)
    # perform feature selection
    _, mid_res = feature_selection(
        X, y;
        groups_separator = groups_separator,

        ex_windows = ex_windows,
        ex_measures = ex_measures,

        extract_tuples = extract_tuples,

        aggrby = (
            # TODO: check correctens: length or length - 2???
            aggrby = tuple(1:length(split(first(names(X)), groups_separator))...),
            aggregatef = aggregatef,
            group_before_score = Val(group_before_score),
        ),

        fs_methods = [ fs_methods...,
            ( # STEP 3: group results by variable
                selector = IdentityFilter(),
                limiter = SoleFeatures.IdentityLimiter(),
            ),
        ],

        fix_special_floats = fix_special_floats,
        fix_special_floats_kwargs = fix_special_floats_kwargs,
        normalize = normalize,
        normalize_kwargs = normalize_kwargs,
        cache_extracted_dataset = cache_extracted_dataset,
        return_mid_results = Val(true)
    )

    # aggregate results to select only variables
    newX = apply_feature_selection_mid_results_to_original_dataset(X, mid_res)

    if isa(return_mid_results, Val{true})
        return (newX, mid_res)
    else
        return newX
    end
end

# ---------------------------------------------------------------------------- #
#                                      debug                                   #
# ---------------------------------------------------------------------------- #
# load a time-series dataset
df, y = SoleData.load_arff_dataset("NATOPS")

ms = [minimum, maximum, mean]
fs_methods = [
	( # STEP 1: unsupervised variance-based filter
		selector = SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
		limiter = SoleFeatures.PercentageLimiter(0.025),
	),
	( # STEP 2: supervised Mutual Information filter
		selector = SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
		limiter = SoleFeatures.PercentageLimiter(0.01),
	),
	( # STEP 3: group results by variable
		selector = SoleFeatures.IdentityFilter(),
		limiter = SoleFeatures.IdentityLimiter(),
	),
]

# prepare dataset for feature selection
Xdf, Xinfo = @test_nowarn SoleFeatures.feature_selection_preprocess(df; features=ms, type=adaptivewindow, nwindows=6, relative_overlap=0.05)
# Xdf, Xinfo = @test_nowarn SoleFeatures.feature_selection_preprocess(df; features=ms, type=wholewindow)

@info "FEATURE SELECTION"

using BenchmarkTools

a = feature_selection(Xdf, y, Xinfo, fs_methods = fs_methods, norm = false)

# 3.189 ms (52904 allocations: 5.54 MiB)
