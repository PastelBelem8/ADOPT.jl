# Dependencies (for reading)
using DelimitedFiles

"Get unique name based on the current datetime"
get_unique_string() = Dates.format(Dates.now(), "yyyymmddHHMMSS")

# Configurations
# File-specific configurations
file_eol = Parameter("\n")
file_sep = Parameter(",")
file_mode = Parameter("a")
file_lang = Parameter(:EN)

# Operations
write_content(method, filename, args...) = let
    @debug "[$(now())][$method] Invoked write on file $(results_file())) in mode $(file_mode()) with values: $(args...)"
    open(filename, file_mode()) do file
        join(file, args..., file_sep())
        write(file, file_eol())
    end
end

write_result(method, args...) = write_content(method, results_file(), vcat(args...))

write_config(method, args...) = with(file_sep, "\n") do
        write_content(method, config_file(), args)
    end

read(method; typ=nothing, header=false) = begin
    @debug "[$(now())][Logging][$method] Invoked read on file $(results_file())) with type $typ and with values: $(args...)"

    open(results_file(), "r") do file
        isnothing(typ) ? readdlm(file, file_sep(), file_eol(); header=header) :
            readdlm(file, file_sep(), typ, file_eol(); header=header)
    end
end

read_matrix(method; header=false) =
    read(method; typ=Float64, header=false)

export write_result, write_config, read, read_matrix

failsafe_mkdir(method, dir) =
    try
        mkdir(dir)
    catch
        @warn "[$(now())][failsafe_mkdir][$method] Could not create $(dir). Ignoring..."
    end
