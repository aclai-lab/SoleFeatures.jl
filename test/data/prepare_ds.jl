using CSV
using DataFrames

# prepare the dataset for testing univariate and multivariate filters
# against the same algos in python and matlab
# this preparation cuts rows with missing values

ds_dir()       = joinpath(dirname(@__FILE__), "csv")
ds(filename)   = joinpath(ds_dir(), filename)
ds_file        = ds("winequalityN.csv")
dataframe      = DataFrame(CSV.File(ds_file))

complete_cases = completecases(dataframe)
complete_df    = dataframe[complete_cases, :]

csv_file       = ds("winequality_complete.csv")
CSV.write(csv_file, complete_df)
