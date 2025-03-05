using SoleFeatures
using Test
using Sole
using Random, StatsBase, DataFrames

# ---------------------------------------------------------------------------- #
#                             DATASET PREPARATION                              #
# ---------------------------------------------------------------------------- #
@testset "feature_selection_preprocess" begin    
    @testset "feature_selection_preprocess DataFrame validation" begin
        X2 = DataFrame(
            temp = [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in 1:5],
            press = [[10.0, 20.0, 30.0, 40.0, 50.0] for _ in 1:5]
        )
        
        # test case 1: Basic functionality with minimal parameters
        @testset "Basic processing" begin
            processed_X, _ = feature_selection_preprocess(
                X2, 
                features = [mean, maximum],
                nwindows = 2
            )
            
            # Check DataFrame structure
            @test size(processed_X, 1) == 5  # Same number of rows
            expected_cols = 2 * 2 * 2  # 2 features × 2 variables × 2 windows
            @test size(processed_X, 2) == expected_cols
            
            # Check column names follow pattern: function(variable)window
            @test any(name -> occursin("mean(temp)w1", name), names(processed_X))
            @test any(name -> occursin("maximum(press)w2", name), names(processed_X))
            
            # Check all values are Float64
            @test all(col -> eltype(col) <: Float64, eachcol(processed_X))
            
            # Check specific computed values (based on our synthetic data)
            mean_temp_col = findfirst(name -> name == "mean(temp)w1", names(processed_X))
            max_press_col = findfirst(name -> name == "maximum(press)w2", names(processed_X))
            
            if !isnothing(mean_temp_col)
                @test all(isapprox.(processed_X[:, mean_temp_col], 2.0, atol=1e-5))
            end
            
            if !isnothing(max_press_col)
                @test all(isapprox.(processed_X[:, max_press_col], 50.0, atol=1e-5))
            end
        end
        
        # Test case 2: Different window types
        @testset "Window types" begin
            # test with wholewindow - should produce just one window
            whole_X, _ = feature_selection_preprocess(
                X2, 
                features = [mean],
                type = SoleFeatures.wholewindow
            )
            
            # Should only have one window per variable/feature
            @test size(whole_X, 2) == 2  # 1 feature × 2 variables × 1 window
            
            # Check prefix format is correct
            @test any(name -> name == "mean(temp)w1", names(whole_X))
            @test any(name -> name == "mean(press)w1", names(whole_X))
            
            # Specifically check that window values are computed correctly
            @test isapprox(whole_X[1, "mean(temp)w1"], 3.0, atol=1e-5)
            @test isapprox(whole_X[1, "mean(press)w1"], 30.0, atol=1e-5)
        end
        
        # Test case 3: Multi-feature test
        @testset "Multiple statistical features" begin
            multi_X, _ = feature_selection_preprocess(
                X2, 
                features = [minimum, mean, maximum],
                nwindows = 1
            )
            
            # Check dimensions
            expected_cols = 3 * 2 * 1  # 3 features × 2 variables × 1 window
            @test size(multi_X, 2) == expected_cols
            
            # Check all three features exist for each variable
            @test any(name -> name == "minimum(temp)w1", names(multi_X))
            @test any(name -> name == "mean(temp)w1", names(multi_X))
            @test any(name -> name == "maximum(temp)w1", names(multi_X))
            
            # Check computed values
            if "minimum(temp)w1" in names(multi_X)
                @test isapprox(multi_X[1, "minimum(temp)w1"], 1.0, atol=1e-5)
            end
            if "maximum(press)w1" in names(multi_X)
                @test isapprox(multi_X[1, "maximum(press)w1"], 50.0, atol=1e-5)
            end
        end
    end

    @testset "InfoFeat metadata validation" begin
        X2 = DataFrame(
            temp = [rand(10) for _ in 1:5],
            press = [rand(10) for _ in 1:5]
        )
        
        # Custom features and windows for predictable results
        custom_features = [mean, std]
        nwin = 3
        vnames = ["temperature", "pressure"]
        
        X, Xinfo = feature_selection_preprocess(X2,
            features = custom_features,
            nwindows = nwin,
            vnames = vnames
        )
        
        # Check that Xinfo is a vector of InfoFeat objects
        @test Xinfo isa Vector{<:SoleFeatures.InfoFeat}
        
        # Check length matches expected count
        expected_count = length(custom_features) * length(vnames) * nwin
        @test length(Xinfo) == expected_count
        @test length(Xinfo) == size(X, 2)  # Should match column count
        
        # Check field values are set correctly
        for (i, f) in enumerate(custom_features)
            for (j, v) in enumerate(vnames)
                for w_idx in 1:nwin
                    # Calculate linear index as done in the function
                    idx = (i-1) * length(vnames) * nwin + (j-1) * nwin + w_idx
                    
                    # Check fields match expected values
                    @test Xinfo[idx].id == idx
                    @test Xinfo[idx].var == v
                    @test Xinfo[idx].feat == Symbol(f)
                    @test Xinfo[idx].nwin == w_idx
                    
                    # Verify column name in X matches InfoFeat metadata
                    expected_name = "$(f)($(v))w$(w_idx)"
                    @test names(X)[idx] == expected_name
                end
            end
        end
    end
end
