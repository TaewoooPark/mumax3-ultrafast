using MicroMagnetic, Metal
set_backend("metal")
MicroMagnetic.set_precision(Float32)

function try_case(f, label)
    try
        f(); println(">>> $label: OK")
    catch e
        println(">>> $label: FAIL")
        m = sprint(showerror, e)
        for ln in split(m, "\n")[1:min(end,6)]; println("      ", ln[1:min(end,110)]); end
    end
end

# public API, exchange only
try_case("public run_until, exchange only") do
    mesh = FDMesh(nx=16, ny=16, nz=1, dx=4e-9, dy=4e-9, dz=4e-9)
    sim = Sim(mesh, driver="LLG", save_data=false)
    set_Ms(sim, 8e5); add_exch(sim, 1.3e-11); init_m0(sim, (1,0.1,0))
    run_until(sim, 1e-12)
end

# public API, with demag, exactly like their std4 example
try_case("public relax(), with demag (their std4 idiom)") do
    mesh = FDMesh(nx=16, ny=16, nz=1, dx=4e-9, dy=4e-9, dz=4e-9)
    sim = Sim(mesh, driver="SD", save_data=false)
    set_Ms(sim, 8e5); init_m0(sim, (1,0.25,0.1)); add_exch(sim, 1.3e-11); add_demag(sim)
    relax(sim; stopping_dmdt=0.01, max_steps=5)
end
