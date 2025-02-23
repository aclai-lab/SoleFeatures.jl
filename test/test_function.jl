# -------------------------------------------------------------------------------------------
# test functions

function build_zero_dim_fake_dataset()
    return DataFrame(:firstCol => [1,1,1,1,1],
                       :secondCol => [1,2,3,4,5],
                       :thirdCol => [70,15,69,80,22],
                       :fourthCol => [10,12,11,10,10])
end

function build_one_dim_fake_dataset()
    return DataFrame(:firstCol => [[0,5],[2,13],[-3,7],[1,-4],[6,0]],
                        :secondCol => [[1,2],[4,5,6],[8,9,6],[10,11,12],[13,14,15]])
end

function build_two_dim_fake_dataset()
    return DataFrame(:firstCol => [ [1 2 3; 4 5 6; 7 8 9], [88 32 18; 15 18 23; 7 8 9], [77 45 36; 8 4 2; 7 3 5] ],
                        :secondCol => [ [1 10 3; 4 5 89; 7 8 9], [88 32 18; 15 18 23; 7 22 9], [77 45 36; 8 33 2; 7 3 5] ])
end

function build_mix_dim_fake_dataset()
    return DataFrame(:firstCol => [ [1 2 3; 4 5 6; 7 8 9], [88 32 18; 15 18 23; 7 8 9], [77 45 36; 8 4 2; 7 3 5] ],
                        :secondCol => [ [1 10 3; 4 5 89; 7 8 9], [88 32 18; 15 18 23; 7 22 9], [77 45 36; 8 33 2; 7 3 5] ])
end

"""
Variance of normalized data:
- firstcol = 0.085
- secondcol = 0.097
- thirdcol = 0.096
- fourthcol = 0.0
- fifthcol = 0.0
"""
function fake_temporal_series_dataset()
    return DataFrame(
        # Variance normalized data: 0.085
        :firstcol => [
            [23.84506, 35.65830, 19.60493, 2.77519, 49.20975, 38.93719, 34.90031, 1.19508, 32.68694, 14.17501],
            [3.74924, 22.20625, 41.62754, 10.84940, 11.22724, 18.88125, 35.72339, 12.30158, 16.84530, 38.72875],
            [7.71919, 31.15404, 29.63745, 21.12171, 17.13501, 2.46123, 33.80773, 36.51729, 28.49147, 25.19532],
            [45.24383, 18.97141, 29.47121, 16.95997, 3.28783, 12.21568, 35.09947, 43.88966, 45.60673, 34.71859],
            [30.47452, 18.58812, 34.30176, 40.25102, 41.88759, 23.09525, 11.02112, 48.12947, 48.67006, 4.52414]
        ],
        # Variance normalized data: 0.097
        :secondcol => [
            [7.27905, 5.30440, 1.46647, 8.76943, 1.40938, 1.35854, 2.77569, 3.96232, 9.00411, 5.07405],
            [9.81052, 3.84083, 1.22610, 8.61556, 1.57618, 5.94720, 6.75046, 4.62765, 4.67943, 1.57160],
            [1.16175, 4.13744, 5.25648, 6.13945, 8.10407, 9.62899, 1.64055, 9.34336, 3.92537, 8.42482],
            [5.46578, 2.40306, 6.27668, 2.83587, 1.31571, 8.86310, 7.32370, 6.58878, 6.99020, 5.51021],
            [4.22164, 6.31536, 4.45163, 1.83739, 7.30354, 3.31067, 1.12208, 7.14623, 7.20354, 1.55776]
        ],
        # Variance normalized data: 0.096
        :thirdcol => [
            [921.37306, 104.69103, 552.51265, 953.41772, 839.79510, 34.38266, 956.11961, 864.40107, 711.73146, 142.60946],
            [424.94985, 383.08780, 901.42079, 224.01437, 238.29513, 728.20146, 507.99875, 860.99781, 823.41544, 602.34972],
            [462.18577, 224.89576, 502.87916, 901.77979, 664.50143, 693.24277, 112.65422, 275.52525, 371.69634, 911.59383],
            [102.30870, 892.44488, 441.16439, 530.80605, 95.69399, 491.99255, 958.48658, 388.28188, 228.27671, 860.60929],
            [498.43672, 510.78201, 221.35961, 559.74567, 757.85110, 831.63683, 233.71335, 932.30539, 661.60417, 427.47744]
        ],
        # Variance normalized data: 0.0
        :fourthcol => [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        ],
        # :A => [
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20)
        # ],
        # :B => [
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20)
        # ],
        # :C => [
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20)
        # ],
        # :D => [
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20)
        # ],
        # :E => [
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20),
        #     rand(20)
        # ],
        # Variance normalized data: 0.0
        :fifthcol => [
            8.0,
            8.0,
            8.0,
            8.0,
            8.0
        ]
    )
end

function build_fake_bit_mask(n=5)::BitVector
    return rand(Bool, n)
end

function random_timeseries_mfd(;ninstances=100, nvar=5, ts_len=5)
    df = random_timeseries_df(;ninstances=ninstances, nvar=nvar, ts_len=ts_len)
    fd = [ collect(1:nvar) ]
    return MultiDataset(df, fd)
end

function random_timeseries_df(;ninstances=100, nvar=5, ts_len=5)
    df = DataFrame()
    for i in 1:nvar
        setproperty!(df, Symbol("a", i), [ rand(ts_len) for _ in 1:ninstances ])
    end
    return df
end

function random_df(;ninstances=100, nvar=5)
    df = DataFrame()
    for i in 1:nvar
        insertcols!(df, string("a", i) => rand(ninstances))
    end
    return df
end

function random_mfd(;ninstances=100, nvar=5)
    df = random_df(;ninstances = ninstances, nvar = nvar)
    fd = [ collect(1:nvar) ]
    return MultiDataset(df, fd)
end
