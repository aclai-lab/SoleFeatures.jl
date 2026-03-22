module SoleFeatures

using  Reexport

# ---------------------------------------------------------------------------- #
#                            sole related packages                             #
# ---------------------------------------------------------------------------- #
@reexport using DataTreatments: aggregate, reducesize
@reexport using DataTreatments: movingwindow, wholewindow, splitwindow, adaptivewindow
using DataTreatments
const DT = DataTreatments

# ---------------------------------------------------------------------------- #
#                              external packages                               #
# ---------------------------------------------------------------------------- #
using DataFrames
using Normalization

# ---------------------------------------------------------------------------- #
#                                 includes                                     #
# ---------------------------------------------------------------------------- #
export load_dataset
export tabular, multidim
export get_data, get_treats
include("dataset.jl")


end