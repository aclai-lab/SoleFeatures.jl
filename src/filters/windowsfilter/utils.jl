# alias: Attribute, MovingWindows, Measure

const AWMDescriptor = Tuple{Symbol, AbstractMovingWindowsIndex, Function}

# constants

const GROUPBY_ATTRIBUTES = :Attributes
const GROUPBY_WINDOWS = :Windows
const GROUPBY_MEASURES = :Measures
const GROUPBY_ID_DICT = Dict{Symbol, Integer}(
    GROUPBY_ATTRIBUTES => 1,
    GROUPBY_WINDOWS => 2,
    GROUPBY_MEASURES => 3
)
const SEPARATOR = "@@"

isdefined(Main, :Catch22) && (Base.nameof(f::SuperFeature) = getname(f)) # wrap for Catch22

function printgroups(groups::Vector{Vector{AWMDescriptor}})
    for (i, g) in enumerate(groups)
        println("Group #$(i):")
        println(join(g, "\n"))
        println()
    end
end

function build_awmds(
    attrs::AbstractVector{Symbol},
    mwies::AbstractVector{<:MovingWindowsIndex},
    measures::AbstractVector{<:Function}
)::Vector{AWMDescriptor}
    return [ Iterators.product(attrs, mwies, measures)... ]
end

function build_awmds(
    attrs::AbstractVector{Symbol},
    mw::AbstractMovingWindows,
    measures::AbstractVector{<:Function}
)::Vector{AWMDescriptor}
    return build_awmds(attrs, [mw...], measures)
end

"""

## Example
```jldoctest
julia> df = DataFrame(:firstcol => [rand(4), rand(4), rand(4)],
                       :secondcol => [rand(4), rand(4), rand(4)])
3×2 DataFrame
 Row │ firstcol                           secondcol
     │ Array…                             Array…
─────┼──────────────────────────────────────────────────────────────────────
   1 │ [0.0197685, 0.915821, 0.37279, 0…  [0.669068, 0.295155, 0.160376, 0…
   2 │ [0.0343932, 0.163887, 0.1159, 0.…  [0.18348, 0.977524, 0.00267274, …
   3 │ [0.769576, 0.627574, 0.634936, 0…  [0.339756, 0.144859, 0.451845, 0…

julia> attrs = Symbol.(names(df))
2-element Vector{Symbol}:
 :firstcol
 :secondcol

julia> fnmw = FixedNumMovingWindows(3, 0.25)
FixedNumMovingWindows(3, 0.25)

julia> measures = [minimum, maximum]
2-element Vector{Function}:
 minimum (generic function with 13 methods)
 maximum (generic function with 13 methods)

julia> awmds = build_awmds(attrs, [ fnmw... ], measures);

julia> expand(df, awmds)
3×12 DataFrame
 Row │ firstcol@@W1(3,0.25)@@minimum  secondcol@@W1(3,0.25)@@minimum  firstcol@@W2(3,0.25)@@minimum  secondcol@@W2(3,0.25)@@minimum  firstcol@@W3(3,0.25)@@minimum  secondcol@@W3(3,0.25)@@minimum  firstcol@@W1(3,0.25)@@maximum  secondcol@@W1(3,0.25)@@maximum  firstcol@@W2(3,0.25)@@maximum  secondc ⋯
     │ Float64                        Float64                         Float64                        Float64                         Float64                        Float64                         Float64                        Float64                         Float64                        Float64 ⋯
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │                     0.0197685                        0.669068                       0.915821                        0.295155                       0.37279                       0.160376                        0.0197685                        0.669068                       0.915821          ⋯
   2 │                     0.0343932                        0.18348                        0.163887                        0.977524                       0.1159                        0.00267274                      0.0343932                        0.18348                        0.163887
   3 │                     0.769576                         0.339756                       0.627574                        0.144859                       0.634936                      0.451845                        0.769576                         0.339756                       0.627574
```
"""
function expand(
    df::AbstractDataFrame,
    awmds::AbstractVector{<:AWMDescriptor}
)::DataFrame
    ndf = DataFrame()
    for ex in awmds
        colname = ex[1]
        insertcols!(ndf, _awm2str(ex) => _buildcol(df[:, colname], ex))
    end
    return ndf
end

function _buildcol(col::AbstractVector, awmd::AWMDescriptor)
    movwin = awmd[2]
    measuref = awmd[3]
    return measuref.([ getwindow(row, movwin) for row in col ])
end

"""
    _awm2str(awmd)

return name to use in expanded DataFrames from AWMDescriptor
"""
function _awm2str(awmd::AWMDescriptor)::String
    attrname = string(awmd[1])
    movwin = string(awmd[2])
    measuref = nameof(awmd[3])
    return join([attrname, movwin, measuref], SEPARATOR)
end

function retrive_groups(
    attributes::AbstractVector{Symbol},
    groupby::Union{Symbol, Tuple{Symbol, Symbol}}
)::Vector{Vector{Symbol}}
    groups = Dict{String, Vector{Symbol}}()
    # from: https://discourse.julialang.org/t/how-can-i-access-multiple-values-of-a-dictionary-using-a-tuple-of-keys/56868/3
    groupids = collect(getindex.(Ref(GROUPBY_ID_DICT), groupby)) # collect(Tuple) -> Vector

    for attr in attributes
        # retrive indicated group 'groupids' from current attribute and check if it belongs to 'groups'
        sel = join(split(String(attr), SEPARATOR)[groupids])
        if (haskey(groups, sel))
            push!(groups[sel], attr)
        else
            groups[sel] = [ attr ]
        end
    end

    return [ values(groups)... ]
end

"""

## Example

```jldoctest
julia> df = DataFrame(:firstcol => [rand(4), rand(4), rand(4)],
                       :secondcol => [rand(4), rand(4), rand(4)])
3×2 DataFrame
 Row │ firstcol                           secondcol
     │ Array…                             Array…
─────┼──────────────────────────────────────────────────────────────────────
   1 │ [0.651899, 0.921898, 0.374045, 0…  [0.302774, 0.244899, 0.0854903, …
   2 │ [0.938466, 0.0454927, 0.168932, …  [0.651345, 0.0348096, 0.669513, …
   3 │ [0.757154, 0.401744, 0.783533, 0…  [0.646735, 0.409603, 0.119752, 0…

julia> attrs = Symbol.(names(df))
2-element Vector{Symbol}:
 :firstcol
 :secondcol

julia> fnmw = FixedNumMovingWindows(3, 0.25)
FixedNumMovingWindows(3, 0.25)

julia> measures = [minimum, maximum]
2-element Vector{Function}:
 minimum (generic function with 13 methods)
 maximum (generic function with 13 methods)

julia> awmds = build_awmds(attrs, [ fnmw... ], measures);

julia> groups = retrive_groups(awmds, :Attributes);

julia> printgroups(groups)
Group #1:
(:firstcol, MovingWindowsIndex{FixedNumMovingWindows}(1, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), minimum)
(:firstcol, MovingWindowsIndex{FixedNumMovingWindows}(2, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), minimum)
(:firstcol, MovingWindowsIndex{FixedNumMovingWindows}(3, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), minimum)
(:firstcol, MovingWindowsIndex{FixedNumMovingWindows}(1, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), maximum)
(:firstcol, MovingWindowsIndex{FixedNumMovingWindows}(2, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), maximum)
(:firstcol, MovingWindowsIndex{FixedNumMovingWindows}(3, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), maximum)

Group #2:
(:secondcol, MovingWindowsIndex{FixedNumMovingWindows}(1, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), minimum)
(:secondcol, MovingWindowsIndex{FixedNumMovingWindows}(2, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), minimum)
(:secondcol, MovingWindowsIndex{FixedNumMovingWindows}(3, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), minimum)
(:secondcol, MovingWindowsIndex{FixedNumMovingWindows}(1, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), maximum)
(:secondcol, MovingWindowsIndex{FixedNumMovingWindows}(2, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), maximum)
(:secondcol, MovingWindowsIndex{FixedNumMovingWindows}(3, Base.RefValue{FixedNumMovingWindows}(FixedNumMovingWindows(3, 0.25))), maximum)
```
"""
function retrive_groups(
    awmds::AbstractVector{<:AWMDescriptor},
    groupby::Union{Symbol, Tuple{Symbol, Symbol}},
)::Vector{Vector{AWMDescriptor}}
    groups = Dict{String, Vector{AWMDescriptor}}()
    # from: https://discourse.julialang.org/t/how-can-i-access-multiple-values-of-a-dictionary-using-a-tuple-of-keys/56868/3
    groupids = [ collect(getindex.(Ref(GROUPBY_ID_DICT), groupby))... ] # collect(Tuple) -> Vector
    for awm in awmds
        # retrive indicated group 'groupids' from current awmd and check if it belongs to 'groups'
        sel = string(awm[groupids]...)
        if (haskey(groups, sel))
            push!(groups[sel], awm)
        else
            groups[sel] = [ awm ]
        end
    end
    return [ values(groups)... ]
end

function evaluate(
    X::AbstractDataFrame,
    y::Union{AbstractVector{<:Union{String, Symbol}}, Nothing},
    awmds::AbstractVector{<:AWMDescriptor},
    selector::AbstractFeaturesSelector,
    groupby::Vector{<:Union{Symbol, Tuple{Symbol, Symbol}}},
    aggregatef::Function,
    limiter::AbstractLimiter;
    normf::Union{Function, Nothing}=nothing,
    normgroup=true,
    supervised=false
)

    @assert length(y)==nrow(X) "Error, non uniform number of instances encountered! $(length(y)) != $(nrow(X))."

    # _a(X) = nothing
    # checks
    if (supervised)
        !is_supervised(selector) && throw(ErrorException("Current selector doesn't contain a supervised implementation"))
        # _a(X) = apply(X, y, selector; returnscores=true)
    else
        !is_unsupervised(selector) && throw(ErrorException("Current selector doesn't contain unsupervised implementation"))
        # _a(X) = apply(X, selector; returnscores=true)
    end
    
    eX = Float64.(expand(X, awmds))

    # global normalization
    if (!isnothing(normf) && !normgroup) eX = normf(eX) end

    for grby in groupby
        groups = retrive_groups(awmds, grby) # groups made of Vector{Vector{AWMDescriptor}}
        groupscores = Vector{Real}(undef, length(groups))

        for (i, grp) in enumerate(groups)
            colsname = _awm2str.(grp) # cols name of the group
            selected_df = eX[:, colsname];

            # group normalization
            if (!isnothing(normf) && normgroup) selected_df = normf(selected_df) end

            # TODO: remove
            # println(selected_df)

            if (supervised)
                _, scores = apply(selected_df, y, selector; returnscores=true)
            else
                _, scores = apply(selected_df, selector; returnscores=true)
            end

            groupscores[i] = aggregatef(scores)
        end

        selectedgroup_idxes = apply_limiter(groupscores, limiter)

        if (iszero(length(selectedgroup_idxes)))
            @warn "No groups respects limiter constraints"
            return Vector{AWMDescriptor}()
        end

        awmds = reduce(vcat, groups[selectedgroup_idxes]) # compact groups
    end
    return awmds
end
