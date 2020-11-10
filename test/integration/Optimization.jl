using Test
using ADOPT

@test begin
    facade(f_length, m_stripes_south03)=parse.(Float64, ["6.7", "11.0", "5.0"])
    apply(obj, 1, 2) == [6.7, 11.0, 5.0]
end
