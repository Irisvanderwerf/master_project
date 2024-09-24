using master_project
using Documenter

DocMeta.setdocmeta!(master_project, :DocTestSetup, :(using master_project); recursive=true)

makedocs(;
    modules=[master_project],
    authors="Irisvanderwerf <i.v.d.werf@student.tue.nl> and contributors",
    sitename="master_project.jl",
    format=Documenter.HTML(;
        canonical="https://Irisvanderwerf.github.io/master_project.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Irisvanderwerf/master_project.jl",
    devbranch="master",
)
