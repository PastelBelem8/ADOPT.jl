# TODO
# 1. Define Clustering techniques
# 2.

function decrease_around(a, b; p)
    Δ = abs(b - a) * 0.8;
    m = (b - a) / 2 + a;
    lb, ub = a, b; p_m = p - m;

    if p > b || p < a
        throw(ArgumentError("invalid point p: cannot decrease around point $p because it does not lie in the range [$a, $b]"))
    end

    if p > m
        ub = min(ub, m + (p-m) + Δ/2)
        lb = ub - Δ
    elseif p < m
        lb = max(lb, m + (p-m) - Δ/2)
        ub = lb + Δ
    else
        lb = m - Δ/2
        ub = m + Δ/2
    end
     lb, ub
end
