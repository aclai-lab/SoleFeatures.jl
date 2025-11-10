using Test
using SoleFeatures

using DataTreatments
using StatsBase

using SoleData.Artifacts
# fill your Artifacts.toml file;
# @test_nowarn fillartifacts()

# ---------------------------------------------------------------------------- #
#                               Prepare Dataset                                #
# ---------------------------------------------------------------------------- #
@info "Prepare Dataset"

natopsloader = NatopsLoader()
X, y = Artifacts.load(natopsloader)

Xdt = DataTreatments.DataTreatment(
    X, :aggregate;
    vnames=names(X),
    win=adaptivewindow(nwindows=3, overlap=0.25),
    features=(DataTreatments.catch22..., minimum, maximum, StatsBase.mean)
)

# ---------------------------------------------------------------------------- #
#                                    Utils                                     #
# ---------------------------------------------------------------------------- #
@info "Utils"

lenvars = length(DataTreatments.get_vnames(Xdt))
lenwins = DataTreatments.get_nwindows(Xdt)
lenmeasures = length(DataTreatments.get_features(Xdt))
lentot = lenvars * lenwins * lenmeasures
println("# Variables: $(lenvars)")
println("# Windows: $(lenwins)")
println("# Measures: $(lenmeasures)")
println("# Total features: $(lentot)")

@test lentot == size(Xdt.dataset, 2)

# ---------------------------------------------------------------------------- #
#                              Feature Selection                               #
# ---------------------------------------------------------------------------- #
@info "Feature Selection"

fs = @test_nowarn SoleFeatures.feature_selection(Xdt, y)

# ---------------------------------------------------------------------------- #
aggrby = (
    aggrby = (:feat,),
    # aggregatef = length, # NOTE: or mean, minimum, maximum to aggregate scores instead of just counting number of selected features for each group
    aggregatef = var,
    group_before_score = true,
)
fs_methods = [
    ( # step 1: unsupervised variance-based filter
        selector = PearsonCorFilter(IdentityLimiter()),
        limiter = PercentageLimiter(0.75),
    ),
    ( # step 2: supervised Mutual Information filter
        selector = MutualInformationClassif(IdentityLimiter()),
        limiter = PercentageLimiter(0.75),
    ),
    ( # step 3: group results by variable
        selector = IdentityFilter(),
        limiter = IdentityLimiter(),
    ),
]
norm::Bool = false

fs = @test_nowarn feature_selection(Xdt, y; aggrby, fs_methods, norm)

# ---------------------------------------------------------------------------- #
aggrby = (
    aggrby = (:nwin,),
    # aggregatef = length, # NOTE: or mean, minimum, maximum to aggregate scores instead of just counting number of selected features for each group
    aggregatef = mean,
    group_before_score = true,
)
fs_methods = [
    ( # step 1: unsupervised variance-based filter
        selector = PearsonCorFilter(IdentityLimiter()),
        limiter = PercentageLimiter(0.75),
    ),
    ( # step 2: supervised Mutual Information filter
        selector = MutualInformationClassif(IdentityLimiter()),
        limiter = PercentageLimiter(0.75),
    ),
    ( # step 3: group results by variable
        selector = IdentityFilter(),
        limiter = IdentityLimiter(),
    ),
]
norm::Bool = true

fs = @test_nowarn feature_selection(Xdt, y; aggrby, fs_methods, norm)
