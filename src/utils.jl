"""
Normalize passed DataFrame using min-max normalization.
Return a new normalized DataFrame
"""
function minmax_normalize(
    df::AbstractDataFrame;
    min_quantile::Float64=0.0,
    max_quantile::Float64=1.0
)::DataFrame
    if min_quantile < 0.0
        throw(DomainError(min_quantile, "min_quantile must be greater than or equal to 0"))
    end
    if max_quantile > 1.0
        throw(DomainError(max_quantile, "max_quantile must be less than or equal to 1"))
    end
    if max_quantile <= min_quantile
        throw(ErrorException("max_quantile must be greater then min_quantile"))
    end

    norm_df = DataFrame()

    for col_name in names(df)
        col = df[:, Symbol(col_name)]
        flatted_col = collect(Iterators.flatten(col))
        dim = SoleBase.dimension(DataFrame(:curr => col))
        tmin = StatsBase.quantile(flatted_col, min_quantile)
        tmax = StatsBase.quantile(flatted_col, max_quantile)
        tmax = 1 / (tmax - tmin)
        dt = StatsBase.UnitRangeTransform(1, 1, true, [tmin], [tmax])

        if dim == 0
            norm_col = StatsBase.transform(dt, Float64.(col))
        elseif dim == 1 || dim == 2
            norm_col = map(r->StatsBase.transform(dt, Float64.(r)),
                Iterators.flatten(eachrow(col)))
        else
            error("unimplemented for dimension >2")
        end

        insertcols!(norm_df, Symbol(col_name) => norm_col)
    end

    return norm_df
end

function minmax_normalize_wrapper(min_quantile::Float64=0.0, max_quantile::Float64=1.0)
    return (df) -> minmax_normalize(
        df,
        min_quantile=min_quantile,
        max_quantile=max_quantile
    )
end
