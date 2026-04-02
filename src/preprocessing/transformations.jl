"""
Data transformations, like Log, Box-Cox and Yeo-Johnson, are mathematical transformations
applied to data to improve its statistical properties. They are especially important in
data analysis and machine learning when dealing with skewed or non-normal data.

Many statistical models (like linear regression) assume that data is approximately normally
distributed, has constant variance, and is not heavily skewed. However, real-world data
often violates these assumptions. Data transformations can help to:
1. Normalize the distribution of the data.
2. Stabilize the variance across the data.
3. Make the data more suitable for modeling and analysis.
"""

"""
Log Transformation
"""
function log_transform(
    vals::AbstractVector{<:Union{Missing,<:Real}};
    base::Real=ℯ,
    offset::Real=0
)
    @assert base > 0 && base != 1 "Base must be > 0 and ≠ 1"

    # Check if all values are > 0
    vals_clean = skipmissing(vals)
    @assert all(v -> v + offset > 0, vals_clean) "All values (after offset) must be > 0 for log"

    return [ismissing(v) ? missing : log(v + offset) / log(base) for v in vals]
end

"""
Box-Cox Transformation
"""
function boxcox_transform(vals::AbstractVector{<:Union{Missing,<:Real}})
    vals_clean = skipmissing(vals) |> collect;
    @assert all(vals_clean .> 0) "Box-Cox requires all values to be strictly positive"
    bc = BoxCox.fit(BoxCoxTransformation, vals_clean)

    return [ismissing(v) ? missing : bc(v) for v in vals]
end

function boxcox_transform(vals::AbstractVector{<:Union{Missing,<:Real}}, λ::Real)
    vals_clean = skipmissing(vals) |> collect;
    @assert all(vals_clean .> 0) "Box-Cox requires all values to be strictly positive"

    return [ismissing(v) ? missing : boxcox(λ, v) for v in vals]
end

"""
Yeo-Johnson Transformation
"""
function yeojohnson_transform(vals::AbstractVector{<:Union{Missing,<:Real}})
    vals_clean = skipmissing(vals) |> collect;
    yj = BoxCox.fit(YeoJohnsonTransformation, vals_clean)

    return [ismissing(v) ? missing : yj(v) for v in vals]
end

function yeojohnson_transform(vals::AbstractVector{<:Union{Missing,<:Real}}, λ::Real)
    return [ismissing(v) ? missing : yeojohnson(λ, v) for v in vals]
end

# ---------------------------------------------------------------------------------------- #
#                                  DataTreatments Interface                                #
# ---------------------------------------------------------------------------------------- #

function log_transform(dt::DataTreatments.DataTreatment; base=ℯ, offset=0)
    dataset, column_names = DataTreatments.get_continuous(dt)

    return Dict(
        col => log_transform(dataset[:, i]; base=base, offset=offset)
        for (i, col) in enumerate(column_names)
    )
end

function boxcox_transform(dt::DataTreatments.DataTreatment)
    dataset, column_names = DataTreatments.get_continuous(dt)

    return Dict(
        col => boxcox_transform(dataset[:, i])
        for (i, col) in enumerate(column_names)
    )
end

function boxcox_transform(dt::DataTreatments.DataTreatment, λ::Real)
    dataset, column_names = DataTreatments.get_continuous(dt)

    return Dict(
        col => boxcox_transform(dataset[:, i], λ)
        for (i, col) in enumerate(column_names)
    )
end

function yeojohnson_transform(dt::DataTreatments.DataTreatment)
    dataset, column_names = DataTreatments.get_continuous(dt)

    return Dict(
        col => yeojohnson_transform(dataset[:, i])
        for (i, col) in enumerate(column_names)
    )
end

function yeojohnson_transform(dt::DataTreatments.DataTreatment, λ::Real)
    dataset, column_names = DataTreatments.get_continuous(dt)

    return Dict(
        col => yeojohnson_transform(dataset[:, i], λ)
        for (i, col) in enumerate(column_names)
    )
end
