using SoleFeatures

using MLJ
using CSV
using DataFrames

ds_dir()    = joinpath(dirname(@__FILE__), "../../data/csv")
ds(filename) = joinpath(ds_dir(), filename)
ds_file = ds("winequality_complete.csv")
dataframe = DataFrame(CSV.File(ds_file))

X = dataframe[:, 2:end]
Xm = Matrix{Float64}(X)
y = dataframe[:, 1]
y = MLJ.levelcode.(categorical(y))

python_dir()    = joinpath(dirname(@__FILE__), "data")
pyt(filename)   = joinpath(python_dir(), filename)

# import pandas as pd
# from mrmr import mrmr_classif

# def main():
#     wine_data = pd.read_csv('/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/test/data/csv/winequality_complete.csv')
#     X = wine_data.iloc[:, 1:]
#     y = wine_data.iloc[:, 0]

#     idx = mrmr_classif(X=X, y=y, K=12, relevance='f', return_scores=False, n_jobs=1)
#     with open("mrmr_f_statistc.txt", "w") as f:
#         for feature in idx:
#             f.write(f"{feature}\n")

# if __name__ == "__main__":
#     main()

pyt_file = pyt("mrmr_f_statistc.txt")
mazzanti_result = readlines(pyt_file)
f_result = mrmr_classif(Xm, y)
f_result = names(X)[f_result]

@test f_result == mazzanti_result

# import pandas as pd
# from mrmr import mrmr_classif

# def main():
#     wine_data = pd.read_csv('/home/paso/Documents/Aclai/Sole/SoleFeatures.jl/test/data/csv/winequality_complete.csv')
#     X = wine_data.iloc[:, 1:]
#     y = wine_data.iloc[:, 0]

#     idx = mrmr_classif(X=X, y=y, K=12, relevance='ks', return_scores=False, n_jobs=1)
#     with open("mrmr_kolmogorov_smirnov.txt", "w") as f:
#         for feature in idx:
#             f.write(f"{feature}\n")

# if __name__ == "__main__":
#     main()

pyt_file = pyt("mrmr_kolmogorov_smirnov.txt")
mazzanti_result = readlines(pyt_file)

ks_result = mrmr_classif(Xm, y; relevance=kolmogorov_smirnov)
ks_result = names(X)[ks_result]

@test ks_result == mazzanti_result

rf_result = mrmr_classif(Xm, y; relevance=random_forest)
rf_result = names(X)[rf_result]