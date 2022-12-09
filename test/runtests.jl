using SoleFeatures
using Test

include("./test_function.jl")

@testset "SoleFeatures.jl" begin

    @testset "transform" begin

        @testset "transform!(mfd, bm; frmidx) using bitmask on a frame of MultiFrameDataset" begin
            df = random_timeseries_df(; nattr=10)
            mfd = SoleBase.MultiFrameDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            bm_frame = BitVector([0,1,0])
            idx_frame = 1
            # expected values
            emfd = deepcopy(mfd)
            SoleBase.dropattributes!(emfd, [3,8])

            transform!(mfd, bm_frame; frmidx=idx_frame)

            @test isequal(mfd, emfd)
        end

        @testset "transform!(mfd, bm) using bitmask on whole MultiFrameDataset" begin
            df = random_timeseries_df(; nattr=5)
            mfd = SoleBase.MultiFrameDataset([[4,2,1], [5,3]], df)
            bm_frame = BitVector([0,1,0,1,1])
            # expected values
            emfd = deepcopy(mfd)
            SoleBase.dropattributes!(emfd, [1,3])

            transform!(mfd, bm_frame)

            @test isequal(mfd, emfd)
        end

        @testset "transform!(df, bm) using bitmask on DataFrame" begin
            df = random_timeseries_df(; nattr=5)
            bm = BitVector([0,1,0,1,1])
            # expected values
            edf = deepcopy(df)
            select!(edf, [2,4,5])

            transform!(df, bm)

            @test isequal(df, edf)
        end

        @testset "transform(mfd, bm; frmidx) using bitmask on a frame of MultiFrameDataset" begin
            df = random_timeseries_df(; nattr=10)
            mfd = SoleBase.MultiFrameDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            bm_frame = BitVector([0,1,0])
            idx_frame = 1
            # expected values
            emfd = deepcopy(mfd)
            SoleBase.dropattributes!(emfd, [3,8])

            mfd = transform(mfd, bm_frame; frmidx=idx_frame)

            @test isequal(mfd, emfd)
        end

        @testset "transform(mfd, bm) using bitmask on whole MultiFrameDataset" begin
            df = random_timeseries_df(; nattr=5)
            mfd = SoleBase.MultiFrameDataset([[4,2,1], [5,3]], df)
            bm_frame = BitVector([0,1,0,1,1])
            # expected values
            emfd = deepcopy(mfd)
            SoleBase.dropattributes!(emfd, [1,3])

            mfd = transform(mfd, bm_frame)

            @test isequal(mfd, emfd)
        end

        @testset "transform(df, bm) using bitmask on DataFrame" begin
            df = random_timeseries_df(; nattr=5)
            bm = BitVector([0,1,0,1,1])
            # expected values
            edf = deepcopy(df)
            select!(edf, [2,4,5])

            df = transform(df, bm)

            @test isequal(df, edf)
        end

    end

    @testset "utils" begin

        @testset "_fr_bm2mfd_bm using array of frames and array of bitmasks" begin
            df = random_timeseries_df(; nattr=10)
            mfd = SoleBase.MultiFrameDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            frms = [1,2,3]
            bms = Vector{BitVector}([ [0,1,0],[0,0,1],[0,1,1,0] ])
            # expected values
            ebm = BitVector([ 0,0,0,1,0,1,1,0,1,0 ])

            resbm = SoleFeatures._fr_bm2mfd_bm(mfd, frms, bms)

            @test isequal(resbm, ebm)
        end

        @testset "_fr_bm2mfd_bm using frame and bitmask" begin
            df = random_timeseries_df(; nattr=10)
            mfd = SoleBase.MultiFrameDataset([[3,7,8], [1,2,4], [5,6,9,10]], df)
            frm = 2
            bm = BitVector([0,0,1])
            # expected values
            ebm = BitVector([ 0,0,1,1,1,1,1,1,1,1 ])

            resbm = SoleFeatures._fr_bm2mfd_bm(mfd, frm, bm)

            @test isequal(resbm, ebm)
        end

    end

    @testset "selectors" begin

        @testset "RandomRanking" begin
            seed = 1997
            rr = RandomRanking(3, seed)
            df = random_timeseries_df(;nattr=10)
            # expected values
            edf = deepcopy(df)
            select!(edf, [1,5,7])

            transform!(df, rr)

            @test isequal(df, edf)
        end

        @testset "VarianceThreshold" begin
            df = fake_temporal_series_dataset()
            ndf = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)
            vt = VarianceThreshold(0.09)
            # expected values
            endf = deepcopy(ndf)
            select!(endf, [2,3])

            transform!(ndf, vt)

            @test isequal(ndf, endf)
        end

        @testset "VarianceRanking" begin
            df = fake_temporal_series_dataset()
            ndf = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)
            vr = VarianceRanking(3)
            # expected values
            endf = deepcopy(ndf)
            select!(endf, [1,2,3])

            transform!(ndf, vr)

            @test isequal(ndf, endf)
        end

        @testset "VarianceRanking on MultiFrameDataset" begin
            df = fake_temporal_series_dataset()
            df = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)
            mfd = SoleBase.MultiFrameDataset([ [1,2,3,4], [5] ], df)
            vr = VarianceRanking(3)
            # expected values
            emfd = deepcopy(mfd)
            SoleBase.dropattributes!(emfd, [4])

            transform!(mfd, vr; frmidx=1)

            @test (isequal(SoleBase.SoleDataset.data(mfd), SoleBase.SoleDataset.data(emfd)) && SoleBase.SoleDataset.frame_descriptor(emfd) == SoleBase.SoleDataset.frame_descriptor(mfd))
        end

    end

end
