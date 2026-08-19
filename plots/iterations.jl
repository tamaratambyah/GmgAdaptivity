using DrWatson
using DataFrames
using Plots
using LaTeXStrings


include("plotools.jl")

# dir = datadir("DarcyAL_Flat_CG" )
dir = datadir("DarcyAL_Flat_GMG_patch" )

df = []
df = collect_results(dir*"/convergence")
sort!(df,[:n])
subset!(df, :γ=> γ -> γ .==1 )

##### External solver iterations
kylov_iters = df[!,:kylov_iters]
u_iters = df[!,:u_iters]
p_iters = df[!,:p_iters]
xplot = df[!,:lvl]


plot()
plot!(xplot,[kylov_iters,u_iters,p_iters],
    lw=2,
    marker=markers[1],
    ms = markersize[1],
    color=colors,
    label=["FMGRES" "velocity" "pressure"])
plot!(shape=:auto,
    # xaxis=:log2,
    # yaxis=:log10,
    xlabel=latexstring("\$ \\ell \$"),
    ylabel="Iterations",
    xtickfontsize=11,ytickfontsize=11,
    xguidefontsize=12,yguidefontsize=12,
    legendfontsize=10,
    legend=:topleft,
    #legend_columns=2,
    framestyle = :box,
    # guidefontfamily=font(20,"Times Roman")
    # ylimits=(1e-9,0)
    )

savefig(dir*"/iterations.pdf")
