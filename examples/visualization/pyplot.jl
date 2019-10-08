#= Pyplot
    Graphics solution for Julia that wrapts pyplot module of matplotlib
    PyPlot uses the Julia PyCall package to call Matplotlib directly from Julia
    with little or no overhead (arrays are passed without making a copy).
=#
using PyPlot
# set a backend
pygui(:tk)
# In general, all of the arguments, including keyword arguments, are exactly
# the same as in Python. (With minor translations, of course, e.g. Julia uses true and nothing instead of Python's True and None.)
x = range(0; stop=2*pi, length=1000); y = sin.(3 * x + 4 * cos.(2 * x));
plot(x, y, color="red", linewidth=2.0, linestyle="--")
title("A sinusoidally modulated sinusoid")
savefig("file.png")
display(gcf())
