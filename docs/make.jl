using SoleFeatures
using Documenter

DocMeta.setdocmeta!(SoleFeatures, :DocTestSetup, :(using SoleFeatures); recursive=true)

makedocs(;
    modules=[SoleFeatures],
    authors="Patrik Cavina", "Federico Manzella", "Giovanni Pagliarini",
    repo="https://github.com/aclai-lab/SoleFeatures.jl/blob/{commit}{path}#{line}",
    sitename="SoleFeatures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aclai-lab.github.io/SoleFeatures.jl",
        assets=String[],
    ),
    pages=[
        "Home"              => "index.md",
        "UnivariateFilters" => "univariate_filters.md"
    ],
)

deploydocs(;
    repo = "github.com/aclai-lab/SoleFeatures.jl",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
