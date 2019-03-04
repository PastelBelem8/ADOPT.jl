#=
    StatsPlot contains statistical recipes for concepts and types
    introduced in the JuliaStats organization
    - > Types (DataFrames, Distributions)
    - > Recipes
        - Histogram / Histogram2d
        - boxplot
        - violin
        - marginalhist
        - corrplot/cornerplot
        - andrewsplot (type of parallel coordinates plot)
        - distribution plots
        - dendograms

    Features ----------------------------------------
    > Visualize a table interactively
    > Seamless plot of IterableTables

    Dependencies ------------------------------------
    - Plots.jl (http://docs.juliaplots.org/latest/)
    - Blink.jl (http://junolab.org/Blink.jl/latest/)
        To install:
            ]add Blink
            using Blink
            Blink.AtomShell.install() # Install dependencies (including Electron)
=#

using StatsPlots
gr(size=(400,300))

using CSVFiles, DataFrames
df = DataFrame(load("results/20190304113552-results.csv"))

# Plot
@df df scatter(:o1, :o2, markersize = 3 / (:var1))

# Filter the dataframe first and then use it to plot
using Query

df |>
    @filter(_.var1 > 0) |>
    # @map({_.b, d = _.c-10}) |>
    @df scatter(:o1, :o2, markersize = 4 .* log.(:var1 .+ 0.1))

# Use Plots grouping capabilities
@df df density([:var1, :o1, :o2], labels=["var1", "o1", "o2"])


# Use blink, visualizing a table interactively
using Blink, Interact
w = Window()
body!(w, dataviewer(df))

import RDatasets
iris = RDatasets.dataset("datasets", "iris")
w = Window()
body!(w, dataviewer(iris))


# Marginal Histograms
@df df marginalhist(:o1, :o2, fc=:plasma, bins=40)

# Corrplot and cornerplot
@df df corrplot(cols(4:6), grid = true, markercolor=:plasma)
@df df cornerplot(cols(4:6), grid = true, markercolor=:plasma, compact=true)

# Box plots and violin plots
@df df violin(:var1, marker=(0.2,:blue,stroke(0)))
@df df boxplot!(:var1, marker=(0.2,:red,stroke(1)))

noisy_df = deepcopy(df)
noisy_df.var1 = noisy_df.var1 .+ 2
@df df violin(:var1, side=:left, marker=(0.2,:blue,stroke(0)), label="Original")
@df noisy_df violin!(:var1, side=:right, marker=(0.2,:orange,stroke(0)), label="Noisy")

# Equal area histograms
@df df ea_histogram(bins = :scott, fillalpha = 0.4)

# Andrewsplot
# AndrewsPlots are a way to visualize structure in high-dimensional data by
# depicting each row of an array or table as a line that varies with the values in columns.
# https://en.wikipedia.org/wiki/Andrews_plot
@df df andrewsplot(:var1, cols(3:4))#, legend = :topleft)

# Grouped Bar Plots (see more information at https://github.com/JuliaPlots/StatsPlots.jl)
groupedbar(rand(10,3), bar_position = :dodge, bar_width=0.7)

# Clustering
using Clustering
D = rand(10, 10)
D += D'
hc = hclust(D, linkage=:single)
plot(hc)

# Distributions
using Distributions
plot(Normal(3,5), fill=(0, .5, :red))

dist = Gamma(2)
scatter(dist, leg=false)
bar!(dist, func=cdf, alpha=0.3)

# Quantile-Quantile plots
x = rand(Normal(), 100)
y = rand(Cauchy(), 100)

plot( qqplot(x, y, qqline = :fit), # qqplot of two samples, show a fitted regression line
 qqplot(Cauchy, y),           # compare with a Cauchy distribution fitted to y; pass an instance (e.g. Normal(0,1)) to compare with a specific distribution
 qqnorm(x, qqline = :R)       # the :R default line passes through the 1st and 3rd quartiles of the distribution
 )
