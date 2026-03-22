# ---------------------------------------------------------------------------- #
#                             data_type callings                               #
# ---------------------------------------------------------------------------- #
tabular = (dt, args...; kwargs...) -> DT.get_tabular(dt, args...; kwargs...)
multidim = (dt, args...; kwargs...) -> DT.get_multidim(dt, args...; kwargs...)

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
function load_dataset(
    dt::DataTreatment,
    args...;
    data_type::Base.Callable=tabular,
    kwargs...
)
    data_type(dt, args...; kwargs...)
end

load_dataset(
    dataset::Matrix,
    vnames::Union{Vector{String},Nothing}=["V$i" for i in 1:size(dataset, 2)],
    target::Union{Nothing,AbstractVector}=nothing,
    args...;
    float_type::Type=Float64,
    kwargs...
) =
    load_dataset(DT.DataTreatment(dataset, vnames, target; float_type), args...; kwargs...)

load_dataset(df::DataFrame, args...; kwargs...) =
    load_dataset(Matrix(df), names(df), args...; kwargs...)