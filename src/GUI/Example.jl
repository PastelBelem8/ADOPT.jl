using Khepri

fixed_xyz_truss_node_family =
  truss_node_family_element(default_truss_node_family(),
                            support=create_node_support("SupportA", ux=true, uy=true, uz=true))

fixed_z_truss_node_family =
  truss_node_family_element(default_truss_node_family(),
                            support=create_node_support("SupportB", ux=false, uy=false, uz=true))

default_truss_bar_family(
  truss_bar_family_element(
    default_truss_bar_family(),
    material=[
      "ElasticIsotropic",   # name
      Khepri.I_MT_STEEL,    # Type
      "Steel",              # Name
      "I'm really steel",   # Nuance
      210000000000.0,       # E
      0.3,                  # NU
      81000000000.0,        # Kirchoff
      77010.0,              # RO
      1.2e-05,              # LX
      0.04,                 # DumpCoef
      235000000.0,          # RE
      360000000.0],         # RT
    section=[
      "Tube",               #name
      "ElasticIsotropic",   #material_name
      false,                #iswood
      [(true,               #solid?
        0.1,                #diameter
        0.01)]]))           #thickness

no_trelica(p) = truss_node(p)

fixed_xyz_no_trelica(p) = truss_node(p, fixed_xyz_truss_node_family)

fixed_z_no_trelica(p) = truss_node(p, fixed_z_truss_node_family)

barra_trelica(p0, p1) = truss_bar(p0, p1)

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#TRELIÇA

nos_trelica(ps) =
  for p in ps
    no_trelica(p)
  end

fixed_nos_trelica(ps) =
  begin
    fixed_xyz_no_trelica(ps[1])
    nos_trelica(ps[2:end-1])
    fixed_xyz_no_trelica(ps[end])
  end

barras_trelica(ps, qs) =
  for (p, q) in zip(ps, qs)
    barra_trelica(p, q)
  end

trelica_espacial(curvas, f, isfirst=true) =
  let (as, bs, cs) = (curvas[1], curvas[2], curvas[3])
    isfirst ? fixed_nos_trelica(as) : nos_trelica(as)
    nos_trelica(bs)
    curvas[4:end] == [] ?
      (fixed_nos_trelica(cs);
       barras_trelica(cs[2:end], cs)) :
      (trelica_espacial(curvas[3:end], f, false);
       barras_trelica(bs, curvas[4]))
    barras_trelica(as, cs)
    barras_trelica(bs, as)
    barras_trelica(bs, cs)
    barras_trelica(bs, as[2:end])
    barras_trelica(bs, cs[2:end])
    barras_trelica(as[2:end], as)
    barras_trelica(bs[2:end], bs)
  end


attractors = Parameter([xyz(5, 5, 5)])
#TRELIÇA
affect_radius(r, p) =
  r*(1+0.5*1.4^(-min(map(attractor -> distance(p, attractor), attractors())...)))

pontos_arco(p, r, fi, psi0, psi1, dpsi) =
  psi0 > psi1 ?
    [] :
    vcat([p+vsph(affect_radius(r, p+vsph(r, fi, psi0)), fi, psi0)],
         pontos_arco(p, r, fi, psi0+dpsi, psi1, dpsi))

coordenadas_trelica_ondulada(p, rac, rb, l, fi, psi0, psi1, dpsi, alfa0, alfa1, d_alfa, d_r) =
  alfa0 >= alfa1 ?
    [pontos_arco(p+vpol(l/2.0, fi-pi/2), rac+d_r*sin(alfa0), fi, psi0, psi1, dpsi)] :
    vcat([pontos_arco(p+vpol(l/2.0, fi-pi/2), rac+d_r*sin(alfa0), fi, psi0, psi1, dpsi)],
         vcat([pontos_arco(p, rb+d_r*sin(alfa0), fi, psi0+dpsi/2, psi1-dpsi/2, dpsi)],
              coordenadas_trelica_ondulada(p+vpol(l, fi+pi/2), rac, rb, l, fi, psi0, psi1, dpsi, alfa0+d_alfa, alfa1, d_alfa, d_r)))

trelica_ondulada(p, rac, rb, l, n, fi, psi0, psi1, alfa0, alfa1, d_alfa, d_r) =
  trelica_espacial(coordenadas_trelica_ondulada(p, rac, rb, l, fi, psi0, psi1, (psi1-psi0)/n, alfa0, alfa1, d_alfa, d_r),
                   panel)

#
using LinearAlgebra
backend(autocad)
# project_kind(Khepri.I_PT_SHELL)

spiked_truss(α1, y1, α2, y2, α3, y3) =
  with(attractors, [sph(10, 0, α1)+vy(y1),
                    sph(10, 0, α2)+vy(y2),
                    sph(10, 0, α3)+vy(y3)]) do
    delete_all_shapes()
    trelica_ondulada(xyz(0, 0, 0),10,9,1.0,10,0,-pi/2,pi/2,0,4*pi,pi/8,0.1)
  end

# spiked_truss(pi/2, 1, pi/3, 3, pi/10, 5)
