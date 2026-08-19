using Plots
_linestyle = [:solid, :dash, :dot, :dashdotdot, :dashdot, :solid, :dash,]
_markers= [:circle, :rect, :star5, :diamond,  :cross, :hexagon]
_colors = palette(:tab10)

markers= [:circle :rect  :diamond ]
markersize = [6 7 6]
linestyle = [:solid :dash]
colors = [_colors[1] _colors[2] _colors[3]]

default(; fontfamily="Computer Modern");
