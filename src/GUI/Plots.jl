using Plots

myplot(solutions::Vector{Solution}) = let
    objs = cat(map(objectives, solutions)...; dims=2)
    plot(objs[1,:], objs[2,:], seriestype=:scatter)
end
