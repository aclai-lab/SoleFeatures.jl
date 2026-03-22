using Test
using SoleFeatures
const SF = SoleFeatures

using DataTreatments
const DT = DataTreatments

using MLJ
using DataFrames, Random
using SoleData: Artifacts

# fill your Artifacts.toml file;
# Artifacts.fillartifacts()

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

Xr, yr = @load_boston
Xr = DataFrame(Xr)

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

a, b = load_dataset(Xc, yc)

a, b =load_dataset(
    Xc, yc,
    TreatmentGroup(name_expr=["petal_length", "petal_width"], grouped=true)
)

a, b =load_dataset(Xr, yr)

a, b =load_dataset(
    Xts, yts,
    TreatmentGroup(
        dims=1,
        aggrfunc=SF.aggregate(
            features=(mean, maximum),
            win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )
    );
    data_type=tabular
)

a, b =load_dataset(
    Xts, yts,
    TreatmentGroup(
        dims=1,
        aggrfunc=SF.reducesize(
            reducefunc=mean,
            win=(splitwindow(nwindows=5),)
        )
    );
    data_type=multidim
)

a, b =load_dataset(
    Xts, yts,
    TreatmentGroup(
        dims=1,
        aggrfunc=SF.reducesize()
    );
    data_type=tabular
)
@test isempty(a)