#= Gadfly (http://gadflyjl.org/stable/tutorial/)
Gadfly is an implementation of a "grammar of graphics" style statistical graphics system for Julia.
=#
using CSVFiles, DataFrames
df = DataFrame(load("results/20190304113552-results.csv"))

using Gadfly
p = plot(df, x=:o1, y=:o2, Geom.point)

# Plot objects can be saved to a file by drawing to one or more backends using draw
img = SVG("ea-optimization.svg", 14cm, 8cm)
draw(img, p)

# We can also display a plot manually using `display`
function get_to_it(d)
  pvo1 = plot(d, x=:var1, y=:o1, Geom.point)
  pvo2 = plot(d, x=:var1, y=:o2, Geom.line)
  pvo1, pvo2
end
ps = get_to_it(df)
map(display, ps)

# Adding other geometries produces layers, which may or may not result in a coherent plot
plot(df, x=:o1, y=:o2, Geom.point, Geom.line)

#=
  You can also work with **ARRAYS**

  Note: With the Array interface, extra elements must be included to specify
  the axis labels, whereas with a DataFrame they default to the column names
=#

var1 = df.var1
o1 = df.o1
o2 = df.o2

plot(x=o1, y=o2, Geom.point, Guide.xlabel("O1"), Guide.ylabel("O2"))

#=
  Change its **aesthetics** by adding some **COLOR**
=#
plot(df, x=:o1, y=:o2, color=:var1, Geom.point, Guide.colorkey(title="Color bar legend :O"))


# SCALE TRANSFORMS
# Scale.x_discrete, Scale.x_log10, Scale.x_continuous...
# Scale.y_discrete, Scale.y_log10, Scale.y_continuous...
using RDatasets
mammals = dataset("MASS", "mammals")

plot(mammals, x=:Body, y=:Brain, label=:Mammal, Geom.point, Geom.label)
plot(mammals, x=:Body, y=:Brain, label=:Mammal, Geom.point, Geom.label,
     Scale.x_log10, Scale.y_log10)


# RENDERING
# Custom graphics library called Compose (primary backend is native SVG)
fig1a = plot(df, x=:o1, y=:o2, Geom.point)
fig1b = plot(df, x=:o1, Geom.bar)
fig1 = hstack(fig1a, fig1b)

# INTERACTIVITY
# One advantage of generating our own SVG is that we can annotate our SVG
# output and embed Javascript code to provide some level of dynamism
gasoline = dataset("Ecdat", "Gasoline")
p = plot(gasoline, x=:Year, y=:LGasPCar, color=:Country, Geom.point, Geom.line)

draw(SVGJS("foo.svg", 100mm, 100mm), p)
