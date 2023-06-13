using HypothesisTests
using StatsBase
using Test
using Revise
using SoleData
using SoleFeatures

include("./test_function.jl")

@testset "SoleFeatures.jl" begin

    @testset "transform" begin

        @testset "transform!(md, bm; frmidx) using bitmask on a frame of MultiModalDataset" begin
            df = random_timeseries_df(; nvar=10)
            md = SoleData.MultiModalDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            bm_frame = BitVector([0,1,0])
            idx_frame = 1
            # expected values
            emd = deepcopy(md)
            SoleData.dropvariables!(emd, [3,8])

            SoleFeatures.transform!(md, bm_frame; frmidx=idx_frame)

            @test isequal(md, emd)
        end

        @testset "transform!(md, bm) using bitmask on whole MultiModalDataset" begin
            df = random_timeseries_df(; nvar=5)
            md = SoleData.MultiModalDataset([[4,2,1], [5,3]], df)
            bm_frame = BitVector([0,1,0,1,1])
            # expected values
            emd = deepcopy(md)
            SoleData.dropvariables!(emd, [1,3])

            SoleFeatures.transform!(md, bm_frame)

            @test isequal(md, emd)
        end

        @testset "transform!(df, bm) using bitmask on DataFrame" begin
            df = random_timeseries_df(; nvar=5)
            bm = BitVector([0,1,0,1,1])
            # expected values
            edf = deepcopy(df)
            select!(edf, [2,4,5])

            SoleFeatures.transform!(df, bm)

            @test isequal(df, edf)
        end

        @testset "transform(md, bm; frmidx) using bitmask on a frame of MultiModalDataset" begin
            df = random_timeseries_df(; nvar=10)
            md = SoleData.MultiModalDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            bm_frame = BitVector([0,1,0])
            idx_frame = 1
            # expected values
            emd = deepcopy(md)
            SoleData.dropvariables!(emd, [3,8])

            md = SoleFeatures.transform(md, bm_frame; frmidx=idx_frame)

            @test isequal(md, emd)
        end

        @testset "transform(md, bm) using bitmask on whole MultiModalDataset" begin
            df = random_timeseries_df(; nvar=5)
            md = SoleData.MultiModalDataset([[4,2,1], [5,3]], df)
            bm_frame = BitVector([0,1,0,1,1])
            # expected values
            emd = deepcopy(md)
            SoleData.dropvariables!(emd, [1,3])

            md = SoleFeatures.transform(md, bm_frame)

            @test isequal(md, emd)
        end

        @testset "transform(df, bm) using bitmask on DataFrame" begin
            df = random_timeseries_df(; nvar=5)
            bm = BitVector([0,1,0,1,1])
            # expected values
            edf = deepcopy(df)
            select!(edf, [2,4,5])

            df = SoleFeatures.transform(df, bm)

            @test isequal(df, edf)
        end

    end

    @testset "utils" begin

        @testset "_fr_bm2md_bm using array of frames and array of bitmasks" begin
            df = random_timeseries_df(; nvar=10)
            md = SoleData.MultiModalDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            frms = [1,2,3]
            bms = Vector{BitVector}([ [0,1,0],[0,0,1],[0,1,1,0] ])
            # expected values
            ebm = BitVector([ 0,0,0,1,0,1,1,0,1,0 ])

            resbm = SoleFeatures._fr_bm2md_bm(md, frms, bms)

            @test isequal(resbm, ebm)
        end

        @testset "_fr_bm2md_bm using frame and bitmask" begin
            df = random_timeseries_df(; nvar=10)
            md = SoleData.MultiModalDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            frm = 2
            bm = BitVector([0,0,1])
            # expected values
            ebm = BitVector([ 0,0,1,1,1,1,1,1,1,1 ])

            resbm = SoleFeatures._fr_bm2md_bm(md, frm, bm)

            @test isequal(resbm, ebm)
        end

    end

    @testset "selectors" begin

        @testset "transform" begin

            @testset "RandomRanking" begin
                seed = 1997
                rr = RandomRanking(3, seed)
                df = random_timeseries_df(;nvar=10)
                # expected values
                edf = deepcopy(df)
                select!(edf, [6,2,5])

                SoleFeatures.transform!(df, rr)

                @test isequal(df, edf)
            end

            @testset "VarianceThreshold" begin
                df = random_df()
                ndf = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)
                vt = VarianceThreshold(0.09)
                @test (SoleFeatures.transform!(df, vt) isa DataFrame)
            end

            @testset "VarianceRanking" begin
                df = random_df()
                ndf = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)
                vr = VarianceRanking(3)
                @test (SoleFeatures.transform!(df, vr) isa DataFrame)
            end

            @testset "StatisticalMajority" begin
                df = random_df()
                y = rand([:a, :b, :c], 100)
                sm = StatisticalMajority(UnequalVarianceTTest)
                @test (SoleFeatures.transform!(df, y, sm) isa DataFrame)
            end

            @testset "StatisticalAtLeastOnce" begin
                df = random_df()
                y = rand([:a, :b, :c], 100)
                sa = StatisticalAtLeastOnce(UnequalVarianceZTest)
                @test (SoleFeatures.transform!(df, y, sa) isa DataFrame)
            end

            @testset "CompoundStatisticalMajority" begin
                df = random_df()
                y = rand([:a, :b, :c], 100)
                cm = CompoundStatisticalMajority(UnequalVarianceTTest, MannWhitneyUTest)
                @test (SoleFeatures.transform!(df, y, cm) isa DataFrame)
            end

            @testset "CompoundStatisticalAtLeastOnce" begin
                df = random_df()
                y = rand([:a, :b, :c], 100)
                ca = CompoundStatisticalAtLeastOnce(UnequalVarianceZTest, MannWhitneyUTest)
                @test (SoleFeatures.transform!(df, y, ca) isa DataFrame)
            end

            @testset "CorrelationFilter" begin
                df = random_df()
                cf = CorrelationFilter(cor, 0)
                @test (SoleFeatures.transform!(df, cf) isa DataFrame)
            end

            @testset "Chi2Filter" begin
                df = random_df()
                y = rand([:a, :b, :c], 100)
                c2r = Chi2Ranking(3)
                @test (SoleFeatures.transform!(df, y, c2r) isa DataFrame)
            end

            @testset "PearsonCorRanking" begin
                df = random_df()
                y = rand(100)
                c2r = PearsonCorRanking(3)
                @test (SoleFeatures.transform!(df, y, c2r) isa DataFrame)
            end

            @testset "MutualInformationClassif" begin
                df = random_df()
                y = rand([:a, :b, :c], 100)
                mir = MutualInformationClassifRanking(3)
                @test (SoleFeatures.transform!(df, y, mir) isa DataFrame)
            end

            # @testset "VarianceRanking on MultiModalDataset" begin
            #     df = random_df();
            #     df = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)
            #     md = SoleData.MultiModalDataset([ [1,2,3,4], [5] ], df)
            #     vr = VarianceRanking(3)
            #     @test (SoleFeatures.transform!(md, vr; frmidx=1) isa MultiModalDataset)
            # end

        end

    end

end
