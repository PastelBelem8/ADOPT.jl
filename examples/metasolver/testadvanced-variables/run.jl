for i in 1:9
    include("t0$(i).jl")
end
for i in 10:26
    include("t$(i).jl")
end
