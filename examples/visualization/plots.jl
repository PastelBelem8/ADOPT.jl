# Provide a considerable comparison between the different graphics library
using Plots
plotlyjs()

plot(sin, (x->begin sin(2x) end), 0, 2π, line=4, leg=false, fill=(0, :orange))

y = rand(100)
plot(0:10:100, rand(11, 4), lab="lines", w=3, palette=:grays, fill=0, α=0.6)
scatter!(y, zcolor=abs.(y .- 0.5), m=(:heat, 0.8, Plots.stroke(1, :green)), ms=10 * abs.(y .- 0.5) .+ 4, lab="grad")

ys = Vector[rand(10), rand(20)]
plot(ys, color=[:black :orange], line=(:dot, 4), marker=([:hex :d], 12, 0.8, Plots.stroke(3, :gray)))
