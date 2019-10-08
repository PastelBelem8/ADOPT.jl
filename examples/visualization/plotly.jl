# Plotly
# Julia interface to the plot.ly plotting library and cloud services
# Plotly capabilities provided by this package are identical to PlotlyJS
# http://spencerlyon.com/PlotlyJS.jl/
using PlotlyJS

my_plot = plot([scatter(x=[1,2], y=[3,4])], Layout(title="My plot"))

## Building Blocks

# Plotly JS figures are constructed by calling the function
# Plotly.newPlot(graphdiv, data, layout)
# where,
#   - graphdiv is an html div element where the plot should appear
#   - data is an array of JSON objects describing the various traces in the visualization
#   - layout is a JSON object describing the layout properties of the visualization.
# Plots have vector of traces

# Three core types
# GenericTrace, Layout, and Plot
function violin_box_overlay()
    y = abs.(100 .* randn(300))
    data = [
        violin(x0="sample 1", name="violin", y=y, points="all"),
        box(x0="sample 1", name="box", y=y, boxpoints=false)
    ]
    plot(data, Layout(legend=attr(x=0.95, xanchor="right")))
end
violin_box_overlay()

# Whenever we click on a point we change its marker symbol to a star and marker color to gold:
# n addition to being able to see our charts in many front-end environments,
# WebIO also provides a 2-way communication bridge between javascript and Julia.
# In fact, when a SyncPlot is constructed, we automatically get listeners for
# all plotly.js javascript events. What's more is that we can hook up Julia
# functions as callbacks when those events are triggered. In the very contrived
# example below we have Julia print out details regarding points on a plot whenever a user hovers over them on the display:
# Add behavior :3
using WebIO
p = plot(rand(10, 4));
display(p)  # usually optional

on(p["hover"]) do data
    println("\nYou hovered over", data)
end

# Change something on the graph!!!
using WebIO
colors = (fill("red", 10), fill("blue", 10))
symbols = (fill("circle", 10), fill("circle", 10))
ys = (rand(10), rand(10))
p = plot(
    [scatter(y=y, marker=attr(color=c, symbol=s, size=15), line_color=c[1])
    for (y, c, s) in zip(ys, colors, symbols)])
display(p)  # usually optional

on(p["click"]) do data
    colors = (fill("red", 10), fill("blue", 10))
    symbols = (fill("circle", 10), fill("circle", 10))
    for point in data["points"]
        colors[point["curveNumber"] + 1][point["pointIndex"] + 1] = "gold"
        symbols[point["curveNumber"] + 1][point["pointIndex"] + 1] = "star"
    end
    restyle!(p, marker_color=colors, marker_symbol=symbols)
end


# Display configuration
p = plot(rand(10, 4), options=Dict(:staticPlot => true))
