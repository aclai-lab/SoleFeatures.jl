# ---------------------------------------------------------------------------- #
#                               Dataset struct                                 #
# ---------------------------------------------------------------------------- #
mutable struct Dataset
    data::Vector{DT.AbstractDataset}
    treats::Vector{DT.TreatmentGroup}
end

# ---------------------------------------------------------------------------- #
#                                Base methods                                  #
# ---------------------------------------------------------------------------- #
Base.length(ds::Dataset) = size(ds.data, 2)
# Base.iterate(ds::Dataset, state=1) =
#     state > length(ds) ? nothing : (ds.data[:, state], state + 1)

# ---------------------------------------------------------------------------- #
#                               getter methods                                 #
# ---------------------------------------------------------------------------- #
get_data(ds::Dataset) = ds.data
get_treats(ds::Dataset) = ds.treats

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
    norm::Union{Type{<:AbstractNormalization},Nothing}=nothing,
    kwargs...
)
    data, treats = data_type(dt, args...; kwargs...)

    if !isnothing(norm)
        for i in eachindex(data)
            grouped = i ≤ length(treats) ? DT.get_grouped(treats[i]) : DT.DefaultGrouped
        end
    end

    Dataset(data, treats)
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