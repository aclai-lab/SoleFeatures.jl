using SoleFeatures
using Documenter

DocMeta.setdocmeta!(SoleFeatures, :DocTestSetup, :(using SoleFeatures); recursive=true)

makedocs(;
    modules=[SoleFeatures],
    authors="Patrik Cavina, Giovanni Pagliarini, Eduard I. Stan",
    repo=Documenter.Remotes.GitHub("aclai-lab", "SoleFeatures.jl"),
    sitename="SoleFeatures.jl",
    format=Documenter.HTML(;
        size_threshold = 4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aclai-lab.github.io/SoleFeatures.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aclai-lab/SoleFeatures.jl",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
