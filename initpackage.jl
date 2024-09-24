using PkgTemplates

name = "master_project"
template = Template(;
    user = "Irisvanderwerf",
    dir = "~/projects",
    julia = v"1",
    plugins = [
        License(; name = "MIT"),
        Git(; ssh = true),
        GitHubActions(; coverage = true),
        Codecov(),
        Documenter{GitHubActions}(; devbranch = "master"),
    ],
)
template(name)