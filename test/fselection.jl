using Test
using SoleFeatures

using DataTreatments

function feature_selection(
    X      :: AbstractArray{T},
    featid :: Vector{DataTreatments.FeatureId},
    y      :: Union{AbstractVector{<:SoleData.SoleBase.CLabel}, Nothing};

    aggrby::Union{ABT,AbstractVector{<:ABT}} = (
        aggrby = (:vname,),
        aggregatef = length, # note: or mean, minimum, maximum to aggregate scores instead of just counting number of selected features for each group
        group_before_score = true,
    ),

    fs_methods::AbstractVector{<:NamedTuple{(:selector, :limiter)}} = [
        ( # step 1: unsupervised variance-based filter
            selector = VarianceFilter(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.1),
        ),
        ( # step 2: supervised Mutual Information filter
            selector = MutualInformationClassif(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.1),
        ),
        ( # step 3: group results by variable
            selector = IdentityFilter(),
            limiter = IdentityLimiter(),
        ),
    ],

    norm::Bool = false,
    normalize_kwargs::NamedTuple = NamedTuple()
) where {T<:Number}
    # prepare aggregation parameters
    if !(aggrby isa AbstractVector)
        # when aggrby is not a Vector assume that the user want to perform aggregation
        #    only during the last step of feature selection TODO: document this properly!!!
        aggrby = push!(Union{Nothing,NamedTuple}[fill(nothing, max(length(fs_methods)-1, 0))...], aggrby)
    end

    # prepare labels
    y_coded = @. CategoricalArrays.levelcode(y)

    # dataset normalization
    norm && _normalize_dataset(X, featid; normalize_kwargs...)

    # feature selection
    fs_mid_results = NamedTuple{(:score, :indices,:group_aggr_func,:group_indices,:aggrby)}[]

    for (fsm, gfs_params) in zip(fs_methods, aggrby)
        current_dataset_col_slice = 1:size(X, 2)

        # pick survived columns only
        for f in fs_mid_results
            current_dataset_col_slice = current_dataset_col_slice[f.indices]
        end

        currX = X[:,current_dataset_col_slice]
        currfeatid = featid[current_dataset_col_slice]

        dataset_param = isnothing(y_coded) || SoleFeatures.is_unsupervised(fsm.selector) ? 
            (currX, currfeatid) : 
            (currX, y_coded, currfeatid)

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

    for f in fs_mid_results
        dataset_col_slice = dataset_col_slice[f.indices]
    end

    namecols = [string(featid[d].feat) * "(" * string(featid[d].vname) * ")w" * string(featid[d].nwin) for d in dataset_col_slice]

    return DataFrame(X[:,dataset_col_slice], namecols), featid[dataset_col_slice], fs_mid_results
end

feature_selection(
    X::DataTreatments.DataTreatment, 
    args...; 
    kwargs...
) = feature_selection(get_dataset(X), get_featureid(X), args...; kwargs...)