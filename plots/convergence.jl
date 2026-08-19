using DrWatson
using DataFrames
using Plots
using LaTeXStrings


include("plotools.jl")

dir = datadir("DarcyAL_Flat")

df = []
df = collect_results(dir*"/convergence")
sort!(df,[:n])
subset!(df, :γ=> γ -> γ .==10 )

xplot = df[(df.p_fe .== 1 ),:lvl]

eu = df[(df.p_fe .== 1 ),:eu_l2]
ep = df[(df.p_fe .== 1 ),:ep_l2]

plot()
plot!(xplot,[eu],
    lw=2,
    marker=markers[1],
    ms = markersize[1],
    color=colors,
    label=[latexstring("\$ p = 1 \$") latexstring("\$ p = 2 \$") latexstring("\$ p = 3 \$")])
plot!(shape=:auto,
    # xaxis=:log2,
    yaxis=:log10,
    xlabel=latexstring("\$ \\ell \$"),
    ylabel=L"$L^2$ error",
    xtickfontsize=11,ytickfontsize=11,
    xguidefontsize=12,yguidefontsize=12,
    legendfontsize=10,
    legend=:bottomleft,
    #legend_columns=2,
    framestyle = :box,
    # guidefontfamily=font(20,"Times Roman")
    # ylimits=(1e-9,0)
    )

savefig(dir*"/convergence.pdf")
