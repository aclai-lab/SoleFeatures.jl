using SoleFeatures
using Test
using Sole
using Random, StatsBase, DataFrames
using MLJTuning

# ---------------------------------------------------------------------------- #
#                             DATASET PREPARATION                              #
# ---------------------------------------------------------------------------- #
X, y       = SoleData.load_arff_dataset("NATOPS")
train_seed = 11
rng        = Random.Xoshiro(train_seed)
Random.seed!(train_seed)

# downsize dataset
num_cols_to_sample = 10
num_rows_to_sample = 50
chosen_cols = StatsBase.sample(rng, 1:size(X, 2), num_cols_to_sample; replace=false)
chosen_rows = StatsBase.sample(rng, 1:size(X, 1), num_rows_to_sample; replace=false)

X = X[chosen_rows, chosen_cols]
y = y[chosen_rows]

ms = [minimum, maximum, mean]

Xdf, Xinfo = @test_nowarn SoleFeatures.feature_selection_preprocess(X; features=ms, nwindows=6)

@testset "Correct values for feature names" begin   
    @testset "minimum(Y[Wrist l])w1" begin
        intervals = 1:10
        stored_result = Xdf[:,1]
        expected_result = minimum.([v[intervals] for v in X[:,1]])
        @test stored_result == expected_result
    end
    @testset "minimum(Z[Elbow l])w4" begin
        intervals = 25:36
        stored_result = Xdf[:,58]
        expected_result = minimum.([v[intervals] for v in X[:,10]])
        @test stored_result == expected_result
    end
    @testset "maximum(X[Thumb r])w6" begin
        intervals = 42:51
        stored_result = Xdf[:,90]
        expected_result = maximum.([v[intervals] for v in X[:,5]])
        @test stored_result == expected_result
    end
    @testset "mean(Z[Hand tip r])w3" begin
        intervals = 16:27
        stored_result = Xdf[:,153]
        expected_result = mean.([v[intervals] for v in X[:,6]])
        @test stored_result == expected_result
    end
end