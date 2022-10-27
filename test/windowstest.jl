using OrderedCollections
using DataFrames
using StatsBase

include("../src/windows/data-filters.jl")
include("../src/windows/windows.jl")
include("../src/windowsfilter/utils.jl")

include("./test_function.jl")

df = random_timeseries_df()
attrname = Symbol.(names(df))
movwin = FixedNumMovingWindows(3, 0.2)
measures = [ minimum, maximum, mean ]
expansions = [ Iterators.product(attrname, movwin, measures)... ]

ndf = expand(df, expansions)
