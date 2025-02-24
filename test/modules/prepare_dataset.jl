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

@testset "feature_selection_preprocess" begin    
    @testset "Basic functionality" begin        
        # Test default parameters
        result = feature_selection_preprocess(X)
        @test result isa DataFrame
        @test all(col -> eltype(col) <: SoleFeatures.Feature, eachcol(result))
        @test size(result, 1) == size(X, 1)
        
        # Test first Feature object properties
        first_feature = first(result[!, 1])
        @test first_feature isa SoleFeatures.Feature
        @test first_feature.var isa String
        @test first_feature.feats isa Symbol
        @test first_feature.nwin isa Int
        @test first_feature.nwin > 0
    end
    
    @testset "Custom parameters" begin
        X2 = DataFrame(
            temp = [rand(10) for _ in 1:5],
            press = [rand(10) for _ in 1:5]
        )
        
        # Custom features and window
        custom_features = [mean, std]
        result = feature_selection_preprocess(X2,
            features = custom_features,
            nwindows = 3,
            vnames = ["temperature", "pressure"]
        )
        
        # Check dimensions
        expected_cols = length(custom_features) * size(X2, 2) * 3  # features * variables * windows
        @test size(result, 2) == expected_cols
        
        # Check feature names
        for (f, v, w) in Iterators.product(custom_features, ["temperature", "pressure"], 1:3)
            col_name = "$(f)($(v))w$(w)"
            @test col_name in names(result)
        end
    end
    
    @testset "Error handling" begin
        # Test with empty DataFrame
        @test_throws ArgumentError feature_selection_preprocess(DataFrame())
        
        # Test with mixed dimensions
        X_invalid = DataFrame(
            a = [1.0, 2.0],
            b = [[1.0, 2.0], [3.0, 4.0]]
        )
        @test_throws DimensionMismatch feature_selection_preprocess(X_invalid)
        
        # Test with invalid windows
        X = DataFrame(a = [rand(10) for _ in 1:5])
        @test_throws ArgumentError feature_selection_preprocess(X, nwindows = 0)
        @test_throws ArgumentError feature_selection_preprocess(X, nwindows = -1)
    end
    
    @testset "Performance" begin
        # Create larger dataset
        X = DataFrame(
            [Symbol("var$i") => [rand(100) for _ in 1:100] for i in 1:5]
        )
        
        # Measure execution time
        time_taken = @elapsed feature_selection_preprocess(X)
        @test time_taken < 5.0  # Should complete within 5 seconds
    end
end
