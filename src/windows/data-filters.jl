# using Logging

# TODO remove this, and replace it with one that varies the number of points per window.
function __moving_window_with_overflow_fixed_num(
	npoints::Integer,
	nwindows::Integer,
	relative_overlap::AbstractFloat,
)::AbstractVector{UnitRange{Int}}

    start = 1
    stop = npoints
    step = (stop-start)/nwindows
    half_context = step*relative_overlap/2
    # println(step)
    # range(start=start, stop=stop, length=nwindows+1)[1:end-1]
    # range(start=start, stop=stop, length=nwindows+1)[1:end-1] .|> (x)->(x+step/2) .|> (x)->(x-size/2,x+size/2)
	# res = [round.(Int, t) for t in range(start=start, stop=stop, length=nwindows+1)[1:end-1] .|> (x)->(x-half_context,x+step+half_context)]
    res = range(start=start, stop=stop, length=nwindows+1)[1:end-1] .|>
		(x)->(x-half_context,x+step+half_context) |>
		t -> round.(Int, t)

	# fix bounds
	res[1] = (max(res[1][1], 1), res[1][2])
	res[end] = (res[end][1], min(res[end][2], npoints))

	_t2r(t::NTuple{2,Int}) = range(t...)

    return _t2r.(res)
end

function __moving_window_without_overflow_fixed_size(
    npoints::Integer,
    window_size::Integer,
    window_step::Integer,
)::AbstractVector{UnitRange{Int}}
    # [w for w in IterTools.partition(1:npoints, window_size, window_step)]
    [i:(i+window_size-1) for i in 1:window_step:(npoints-(window_size-1))]
end

function __moving_window_without_overflow_fixed_size(
    npoints::Integer,
    window_size::AbstractFloat,
    window_step::AbstractFloat,
)::AbstractVector{UnitRange{Int}}
    window_size = round(Int, window_size) # TODO maybe floor?
    # [clamp(round(Int, i), 1, npoints):clamp(round(Int, i)+window_size-1, 1, npoints) for i in 1:window_step:(npoints-(window_size-1))]
    [round(Int, i):round(Int, i)+window_size-1 for i in 1:window_step:(npoints-(window_size-1))]
end

function __moving_window_without_overflow_fixed_num(
	npoints::Integer,
	nwindows::Integer,
	relative_overlap::AbstractFloat,
)::AbstractVector{UnitRange{Int}}
	# start = 1+half_context
	# stop = npoints-half_context
	# step = (stop-start+1)/nwindows
	# half_context = step*relative_overlap/2

	# half_context = relative_overlap * (npoints-1) / (2* nwindows+2*relative_overlap)
	half_context = relative_overlap * npoints / (2* nwindows+2*relative_overlap)
	start = 1+half_context
	stop = npoints-half_context
	step = (stop-start+1)/nwindows

	# _w = floor(Int, step+2*half_context)
	# _w = floor(Int, ((stop-start+1)/nwindows)+2*half_context)
	# _w = floor(Int, ((npoints-half_context)-(1+half_context)+1)/nwindows+2*half_context)
	# _w = floor(Int, (npoints-2*half_context)/nwindows+2*half_context)
	_w = floor(Int, (npoints-2*half_context)/nwindows + 2*half_context)

	# println("step: ($(stop)-$(start)+1)/$(nwindows) = ($(stop-start+1)/$(nwindows) = $(step)")
	# println("half_context: $(half_context)")
	# first_points = range(start=start, stop=stop, length=nwindows+1)[1:end-1]
	first_points = range(start=start, stop=stop, length=nwindows+1)[1:end-1] # TODO needs Julia 1.7: warn user
	first_points = map((x)->x-half_context, first_points)
	@assert isapprox(first_points[1], 1.0)
	# println("first_points: $(collect(first_points))")
	# println("window: $(step)+$(2*half_context) = $(step+2*half_context)")
	# println("windowi: $(_w)")
	first_points = map((x)->round(Int, x), first_points)
	# first_points .|> (x)->(x+step/2) .|> (x)->(x-size/2,x+size/2)
	# first_points .|> (x)->(max(1.0,x-half_context),min(x+step+half_context,npoints))
	# first_points .|> (x)->(x-half_context,x+step+half_context)
	first_points .|> (xi)->(xi:xi+_w-1)
end


# For DEBUGGING:
# Problematic:
# _moving_window(200, 2, 2.0)
# _moving_window(60, 19, 0) |> collect
# _moving_window(60, 19, 1/10) |> collect
# _moving_window(60, 19, 1/3) |> collect
# _moving_window(60, 19, 1/4) |> collect
# _moving_window(60, 19, 1/5) |> collect

############################################################################################
"""
	_moving_window(npoints, nwindows, relative_overlap; allow_overflow = false)

Get the ranges of indices to apply a moving window to a Vector of length `npoints` to divide
it in `nwindows` windows with a relative overlap of `relative_overlap`.

If `allow_overflow` is set to `true` then the length of each window may be different from
one another.

	_moving_window(vec, nwindows, relative_overlap; allow_overflow = false)

Get the ranges of indices to apply a moving window to the Vector `vec` to divide
it in `nwindows` windows with a relative overlap of `relative_overlap`.
"""
# Moving window method for enumerating windows

function _moving_window(
	npoints::Integer;
    #
    nwindows::Union{Nothing,Integer} = nothing,
    relative_overlap::Union{Nothing,AbstractFloat} = nothing,
    #
    window_size::Union{Nothing,Number} = nothing,
    window_step::Union{Nothing,Number} = nothing,
    #
    allow_overflow::Bool = false,
    kwargs...
)
    fixed_num_mode  = !isnothing(nwindows)    && !isnothing(relative_overlap)
    fixed_size_mode = !isnothing(window_size) && !isnothing(window_step)
    if !fixed_num_mode && !fixed_size_mode
        return [1:npoints]
    end
    _partial_fixed_num_mode  = !isnothing(nwindows)    || !isnothing(relative_overlap)
    _partial_fixed_size_mode = !isnothing(window_size) || !isnothing(window_step)
    @assert !(fixed_num_mode && fixed_size_mode) && (!fixed_size_mode || !_partial_fixed_num_mode) && (!fixed_num_mode || !_partial_fixed_size_mode) "A moving_window-based filter requires exactly two parameters: either `window_size` & `window_step` or `nwindows` & `relative_overlap`."

	if fixed_num_mode
        @assert relative_overlap ≥ 0 "`relative_overlap` must be greater or equal to 0 ($relative_overlap)"
        if nwindows >= npoints
            @warn "Warning! Performing moving window with nwindows >= npoints: $(nwindows) >= $(npoints)"
        end
        _ma_fun = allow_overflow ? __moving_window_with_overflow_fixed_num : __moving_window_without_overflow_fixed_num
        # _ma_fun(npoints; nwindows = nwindows, relative_overlap = relative_overlap, kwargs...)
        _ma_fun(npoints, nwindows, relative_overlap; kwargs...)
    else
        if window_size >= npoints
            @warn "Warning! Performing moving window with window_size >= npoints: $(window_size) >= $(npoints)"
        end
        window_size = min(npoints, window_size)
        _ma_fun = allow_overflow ? __moving_window_with_overflow_fixed_size : __moving_window_without_overflow_fixed_size
        # _ma_fun(npoints; window_size, window_step, kwargs...)
        _ma_fun(npoints, window_size, window_step; kwargs...)
    end
end

# TODO defaults to nwindows+relativo_overlap mode
# function _moving_window(
#     npoints::Integer,
#     nwindows::Integer,
#     relative_overlap::AbstractFloat,
#     args...;
#     kwargs...
# )
#     _moving_window(npoints, nwindows = nwindows, relative_overlap = relative_overlap, args...; kwargs...)
# end

# Moving window method for enumerating window contents
function _moving_window(univariate_series::AbstractVector{T}, args...; return_dict = false, kwargs...) where {T}
    dict = OrderedDict([r => univariate_series[r] for r in _moving_window(length(univariate_series), args...; kwargs...)])
    return_dict ? dict : collect(values(dict))
end
function _moving_window(multivariate_series::AbstractMatrix{T}, args...; return_dict = false, kwargs...) where {T}
    dict = OrderedDict([r => multivariate_series[r,:] for r in _moving_window(size(multivariate_series, 1), args...; kwargs...)])
    return_dict ? dict : collect(values(dict))
end
function _moving_window(multivariate_series_dataset::Union{AbstractArray{T,3},D}, args...; return_dict = false, kwargs...) where {T, D<:OrderedDict{<:Any, <:AbstractMatrix{T}}}
    dict = OrderedDict(Iterators.flatten([begin
        instance_windows_dict = _moving_window(instance, args...; return_dict = true, kwargs...)
        OrderedDict(zip(map((k)->(i_instance,k), collect(keys(instance_windows_dict))), values(instance_windows_dict)))
    end for (i_instance, instance) in enumerate_instances(multivariate_series_dataset)]))
    return_dict ? dict : cat(values(dict)..., dims=3)
end

############################################################################################
# Moving window filter

function moving_window_filter(
    univariate_series::AbstractVector{T};
    f::Function,
    kwargs...
)::AbstractVector{T} where {T}
    map(f, _moving_window(univariate_series; return_dict = false, kwargs...))
end

function moving_window_filter(
    multivariate_series::AbstractMatrix{T};
    f::Function,
    kwargs...
)::Matrix{T} where {T}
    hcat(map((w)->(f.(eachcol(w))), _moving_window(multivariate_series; return_dict = false, kwargs...))...)'
end

function moving_window_filter(
    multivariate_series_dataset::AbstractArray{T,3};
    f::Function,
    kwargs...
)::Array{T,3} where {T}
    cat([moving_window_filter(instance; f = f, kwargs...) for (i_instance, instance) in enumerate_instances(multivariate_series_dataset)]..., dims=3)
end

function moving_window_filter(
    multivariate_series_dataset::D;
    f::Function,
    kwargs...
)::D where {T,ID,D<:OrderedDict{ID, <:AbstractMatrix{T}}}
    OrderedDict([i_instance => moving_window_filter(instance; f = f, kwargs...) for (i_instance, instance) in enumerate_instances(multivariate_series_dataset)])
end

# Moving window filter
function moving_average_filter(
    X::Any,
    nwindows::Integer,
    relative_overlap::AbstractFloat;
    kwargs...
)
    moving_window_filter(X; f = StatsBase.mean, nwindows = nwindows, relative_overlap = relative_overlap, kwargs...)
end
function moving_average_filter(
    X::Any;
    kwargs...
)
    moving_window_filter(X; f = StatsBase.mean, kwargs...)
end


# TODO remove these old functions
moving_average(univariate_series::AbstractVector, n::Nothing, st::Nothing) = univariate_series
moving_average(univariate_series::AbstractVector, n::Integer, st::Integer) = [sum(@view univariate_series[idxs])/n for idxs in __moving_window_without_overflow_fixed_size(length(univariate_series), n, st)]
moving_average(multivariate_series::AbstractMatrix, n::Integer, st::Integer) = mapslices((x)->(@views moving_average(x,n,st)), multivariate_series, dims=1)
moving_average_same(univariate_series::AbstractVector, n::Integer)  = [StatsBase.mean(@view univariate_series[max((i-n),1):min((i+n),length(univariate_series))]) for i in 1:length(univariate_series)]
function moving_average(multivariate_series_dataset::AbstractArray{T,3},  n::Integer, st::Integer) where {T}
    n_points, n_attributes, n_instances = size(multivariate_series_dataset)
    new_n_points = length(moving_average(Vector{Int}(undef, n_points), n, st))
    new_X = similar(multivariate_series_dataset, (new_n_points, n_attributes, n_instances))
    for n_instance in 1:n_instances
        new_instance = moving_average(multivariate_series_dataset[:, :, n_instance], n, st)
        new_X[:, :, n_instance] .= new_instance
    end
    return new_X
end

############################################################################################
# Moving window partitioning filter
function moving_partitioning_filter(
    multivariate_series::AbstractMatrix{T},
    args...;
    return_dict = true,
    kwargs...
) where {T}
    dict = _moving_window(multivariate_series, args...; return_dict = true, kwargs...)
    return_dict ? dict : cat(values(dict)..., dims=3)
end

function moving_partitioning_filter(
    multivariate_series_dataset::AbstractArray{T,3},
    args...;
    return_dict = true,
    kwargs...
) where {T}
    dict = _moving_window(multivariate_series_dataset, args...; return_dict = true, kwargs...)
    return_dict ? dict : cat(values(dict)..., dims=3)
end

function moving_partitioning_filter(
    multivariate_series_dataset::OrderedDict{ID, <:AbstractMatrix{T}},
    args...;
    return_dict = true,
    kwargs...
)::OrderedDict{<:Tuple{ID,<:UnitRange{<:Int}}, <:AbstractMatrix{T}} where {T,ID}
    dict = _moving_window(multivariate_series_dataset, args...; return_dict = true, kwargs...)
    return_dict ? dict : cat(values(dict)..., dims=3)
end
