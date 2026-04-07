module SoleFeatures

# ---------------------------------------------------------------------------- #
#                            sole related packages                             #
# ---------------------------------------------------------------------------- #
# @reexport using DataTreatments: aggregate, reducesize
# @reexport using DataTreatments: movingwindow, wholewindow, splitwindow, adaptivewindow
export aggregate, reducesize
export movingwindow, wholewindow, splitwindow, adaptivewindow
using DataTreatments
const DT = DataTreatments

# ---------------------------------------------------------------------------- #
#                              external packages                               #
# ---------------------------------------------------------------------------- #
using DataFrames
using StatsBase
using Normalization

# ---------------------------------------------------------------------------- #
#                                 includes                                     #
# ---------------------------------------------------------------------------- #
export load_dataset
export tabular, multidim
export get_data, get_treats
include("dataset.jl")

import Normalization: @_Normalization
export ZScore, MinMax, Center, Sigmoid, UnitEnergy, UnitPower
export Scale, ScaleMad, ScaleFirst, PNorm1, PNorm, PNormInf
export Robust
include("normalization.jl")


end