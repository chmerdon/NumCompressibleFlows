using NumCompressibleFlows
using ExtendableFEM
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize
using Symbolics
using LinearAlgebra
#using Test #hide

# new packages
using DrWatson
using JLD2
using LaTeXStrings
using Colors
using ColorTypes
#gr()

quickactivate(@__DIR__, "NumCompressibleFlows")
mkpath(plotsdir("compressible_stokes/convegence_history"))

default_args = Dict(
    # problem parameters
    "μ" => 1,
    "λ" => 0,
    "γ" => 1,
    "c" => 1,
    "M" => 1,
    # solving options
    "τfac" => 1,
    "ufac" => 1000,
    "nrefs" => 1,
    "order" => 1,
    "pressure_stab" => 0,
    "maxsteps" => 5000,
    "target_residual" => 1.0e-11,
    "reconstruct" => true,
    # data of the problem
    "velocitytype" => ZeroVelocity,
    "densitytype" => ExponentialDensity,
    "eostype" => IdealGasLaw,
    "gridtype" => Mountain2D,
    "pressure_in_f" => false,
    "laplacian_in_rhs" => true,

)

function load_data(; kwargs...)
    data = deepcopy(default_args) 
    for (k,v) in kwargs 
        data[String(k)] = v 
    end
    return data
end


function plot_convergencehistory(; nrefs = 1:6, Plotter = Plots, force = false, kwargs...)

    data = load_data(; kwargs...)
    @show data
    Results = zeros(Float64, length(nrefs), 5)
    NDoFs = zeros(Float64, length(nrefs))

    for lvl in nrefs
        data["nrefs"] = lvl
        data, ~ = produce_or_load(run_single, data, filename = filename, force = force)
        NDoFs[lvl] = data["ndofs"]
        Results[lvl,1] = data["Error(L2,u)"]
        Results[lvl,2] = data["Error(H1,u)"] 
        Results[lvl,3] = data["Error(L2,ϱ)"]
        Results[lvl,4] = data["Error(L2,ϱu)"]
        Results[lvl,5] = data["nits"]

        print_convergencehistory(NDoFs[1:lvl], Results[1:lvl, :]; X_to_h = X -> X .^ (-1 / 2), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||", "|| ϱ - ϱ_h ||", "|| ϱu - ϱu_h ||", "#its"], xlabel = "ndof")
    end

    ## plot
    #Plotter.rc("font", size=20)
    yticks = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2]
    xticks = [10,1e2,1e3,1e4,1e5]
    Plotter.plot(; show = true, size = (1600,1000), margin = 1Plots.cm, legendfontsize = 20, tickfontsize = 22, guidefontsize = 26, grid=true)
    Plotter.plot!(NDoFs, Results[:,2]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| ∇(\mathbf{u} - \mathbf{u}_h)\,||", grid=true)
    Plotter.plot!(NDoFs, Results[:,3]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ}-ϱ_h \, ||", grid=true)
    Plotter.plot!(NDoFs, Results[:,4]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| {ϱ\mathbf{u}}-ϱ_h \mathbf{u}_h \, ||", grid=true)
    Plotter.plot!(NDoFs, Results[:,1]; xscale = :log10, yscale = :log10, linewidth = 3, marker = :circle, markersize = 5, label = L"|| \mathbf{u} - \mathbf{u}_h \,||", grid=true)
    Plotter.plot!(NDoFs, 200*NDoFs.^(-0.5); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h)", grid=true)
    Plotter.plot!(NDoFs, 200*NDoFs.^(-1.0); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h^2)", grid=true)
    Plotter.plot!(NDoFs, 100*NDoFs.^(-1.25); xscale = :log10, yscale = :log10, linestyle = :dash, linewidth = 3, color = :gray, label = L"\mathcal{O}(h^{2.5})", grid=true)
    
    Plotter.plot!(; legend = :bottomleft, xtick = xticks, yticks = yticks, ylim = (yticks[1]/2, 2*yticks[end]), xlim = (xticks[1], xticks[end]), xlabel = "ndofs",gridalpha = 0.7,grid=true, background_color_legend = RGBA(1,1,1,0.7))
    ## save
    Plotter.savefig("Aconvegence_history.png")
end
function main(;
    nrefs = 4,
    M = 1,
    c = 1,
    μ = 1,
    λ = -2*μ / 3,
    ufac = 1,
    τfac = 1,
    order = 1,
    pressure_stab = 0,
    pressure_in_f = true, # default is well-balancedness
    laplacian_in_rhs = true, # default everything in the force (f or g)
    conv_parameter = 0,
    velocitytype = ZeroVelocity, 
    densitytype = ExponentialDensity,
    eostype = IdealGasLaw,
    gridtype = Mountain2D,
    bonus_quadorder = 4,
    bonus_quadorder_f = bonus_quadorder,
    bonus_quadorder_g = bonus_quadorder,
    bonus_quadorder_bnd = bonus_quadorder,
    maxsteps = 5000,
    target_residual = 1.0e-11,
    Plotter = nothing,
    reconstruct = true,
    γ=1,
    kwargs...
)

## load data for testcase
#grid_builder, kernel_gravity!, kernel_rhs!, u!, ∇u!, ϱ!, τfac = load_testcase_data(testcase; laplacian_in_rhs = laplacian_in_rhs,Akbas_example=Akbas_example, M = M, c = c, μ = μ,γ=γ, ufac = ufac)
ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! = prepare_data(velocitytype, densitytype, eostype; laplacian_in_rhs = laplacian_in_rhs, pressure_in_f = pressure_in_f, M = M, c = c, μ = μ, λ = λ,γ=γ, ufac = ufac, conv_parameter =conv_parameter )

xgrid = NumCompressibleFlows.grid(gridtype; nref = 4)
M_exact = integrate(xgrid, ON_CELLS, ϱ!, 1; quadorder = 30) 
M = M_exact
 τ = μ / (c*order^2 * M * sqrt(τfac)*ufac) # time step for pseudo timestepping
 #τ = μ / (4*order^2 * M * sqrt(τfac)) 
@info "M = $M, τ = $τ"
sleep(1)

## define unknowns
u = Unknown("u"; name = "velocity", dim = 2)
ϱ = Unknown("ϱ"; name = "density", dim = 1)
p = Unknown("p"; name = "pressure", dim = 1)

## define reconstruction operator
if order == 1
    FETypes = [H1BR{2}, L2P0{1}, L2P0{1}] # H1BR Bernardi-Raugel 2 is the dimension, L2P0 is P0 finite element
    id_u = reconstruct ? apply(u, Reconstruct{HDIVRT0{2}, Identity}) : id(u)# if reconstruct is true call apply, if false call id
    div_u = reconstruct ? apply(u, Reconstruct{HDIVRT0{2}, Divergence}) : div(u) # Marwa div term 
    # RT of lowest order reconstruction 
elseif order == 2
    FETypes = [H1P2B{2, 2}, L2P1{1}, L2P1{1}] #H1P2B add additional cell bubbles, not Bernardi-Raugel? L2P1 is P1 finite element
    id_u = reconstruct ? apply(u, Reconstruct{HDIVRT1{2}, Identity}) : id(u) # RT of order 1 reconstruction
    div_u = reconstruct ? apply(u, Reconstruct{HDIVRT1{2}, Divergence}) : div(u) # Marwa div term 
    
end

## in/outflow regions
testgrid = NumCompressibleFlows.grid(gridtype; nref = 1)
rinflow = inflow_regions(velocitytype, gridtype)
routflow = outflow_regions(velocitytype, gridtype)
rhom = setdiff(unique!(testgrid[BFaceRegions]), union(rinflow,routflow))
@info rinflow, routflow, rhom
sleep(1)

## define first sub-problem: Stokes equations to solve for velocity u
PD = ProblemDescription("Stokes problem")
assign_unknown!(PD, u)
assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
assign_operator!(PD, BilinearOperator([div_u]; factor = λ, store = true, kwargs...)) # Marwa div term 
if conv_parameter > 0
    assign_operator!(PD, LinearOperator(kernel_convection_linearoperator!, [
    id_u], [id(u),grad(u),id(ϱ)]; quadorder = 2*order + 1, factor = -1, kwargs...))
end

# Start adding hom boundary data 
assign_operator!(PD, LinearOperator(eos!(eostype), [div(u)], [id(ϱ)]; factor = c, kwargs...))
if length(rhom) > 0 
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = rhom, kwargs...))
end
if length(rinflow) > 0 || length(routflow) > 0
    assign_operator!(PD, InterpolateBoundaryData(u, u!; bonus_quadorder = bonus_quadorder_bnd, regions = union(rinflow, routflow), kwargs...))
end

# 
if kernel_rhs! !== nothing
    assign_operator!(PD, LinearOperator(kernel_rhs!, [id_u]; factor = 1, store = true, bonus_quadorder = bonus_quadorder_f, kwargs...))
end
assign_operator!(PD, LinearOperator(kernel_gravity!, [id_u], [id(ϱ)]; factor = 1, bonus_quadorder = bonus_quadorder_g, kwargs...))

## FVM for continuity equation
@info "timestep = $τ"
PDT = ProblemDescription("continuity equation")
assign_unknown!(PDT, ϱ)
if order > 1
    assign_operator!(PDT, BilinearOperator(kernel_continuity!, [grad(ϱ)], [id(ϱ)], [id(u)]; quadorder = 2 * order, factor = -1, kwargs...))
    # entities = ON_CELLS since it is the default
end
if pressure_stab > 0
    psf = pressure_stab #* xgrid[CellVolumes][1]
    assign_operator!(PDT, BilinearOperator(stab_kernel!, [jump(id(ϱ))], [jump(id(ϱ))], [id(u)]; entities = ON_IFACES, factor = psf, kwargs...))
end
assign_operator!(PDT, BilinearOperator([id(ϱ)]; quadorder = 2 * (order - 1), factor = 1 / τ, store = true, kwargs...)) # for (1/τ) (ϱ^n+1,λ_h)
assign_operator!(PDT, LinearOperator([id(ϱ)], [id(ϱ)]; quadorder = 2 * (order - 1), factor = 1 / τ, kwargs...)) # for (1/τ) (ϱ^n,λ_h) on the rhs 
assign_operator!(PDT, BilinearOperatorDG(kernel_upwind!, [jump(id(ϱ))], [this(id(ϱ)), other(id(ϱ))], [id(u)]; factor = 1, quadorder = order+1, entities = ON_IFACES, kwargs...))
# [jump(id(ϱ))] is the test function [λ]
# [jump(id(ϱ))]is test function lambda , [this(id(ϱ)), other(id(ϱ))] is the the flux multlplied by lambda_upwind. [id(u)] is the function u that is needed
# Start adding inflow 
if length(rinflow) > 0
    assign_operator!(PDT, LinearOperatorDG(kernel_inflow!(u!,ϱ!), [id(ϱ)]; factor = -1, bonus_quadorder = bonus_quadorder_bnd, entities = ON_BFACES, regions = rinflow, kwargs...))    
end
if length(routflow) > 0
    #assign_operator!(PDT, LinearOperatorDG(kernel_inflow!(u!,ϱ!), [id(ϱ)]; factor = -1, bonus_quadorder = bonus_quadorder_bnd, entities = ON_BFACES, regions = routflow, kwargs...))    
    assign_operator!(PDT, BilinearOperator(kernel_outflow!(u!), [id(ϱ)]; factor = 1, bonus_quadorder = bonus_quadorder_bnd, entities = ON_BFACES, regions = routflow, kwargs...))    
end
   

## prepare error calculation
EnergyIntegrator = ItemIntegrator(energy_kernel!, [id(u)]; resultdim = 1, quadorder = 2 * (order + 1), kwargs...)
ErrorIntegratorExact = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)]; resultdim = 9, quadorder = 2 * (order + 1), kwargs...)
MassIntegrator = ItemIntegrator([id(ϱ)]; resultdim = 1, kwargs...)
NDofs = zeros(Int, nrefs)
Results = zeros(Float64, nrefs, 5) # it is a matrix whose rows are levels and columns are 

sol = nothing
xgrid = nothing
op_upwind = 0
for lvl in 1:nrefs
    xgrid = NumCompressibleFlows.grid(gridtype; nref = lvl)
    @show xgrid
    FES = [FESpace{FETypes[j]}(xgrid) for j in 1:3] # 3 because we have dim(FETypes)=3
    sol = FEVector(FES; tags = [u, ϱ, p]) # create solution vector and tag blocks with the unknowns (u,ρ,p) that has the same order as FETypes

    ## initial guess
    fill!(sol[ϱ], M) # fill block corresponding to unknown ρ with initial value M, in Algorithm it is M/|Ω|?? We could write it as M/|Ω| and delete area from down there
    interpolate!(sol[u], u!)
    interpolate!(sol[ϱ], ϱ!)
    NDofs[lvl] = length(sol.entries)


    M = sum(evaluate(MassIntegrator, sol))

    ## solve the two problems iteratively [1] >> [2] >> [1] >> [2] ...
    SC1 = SolverConfiguration(PD; init = sol, maxiterations = 1, target_residual = target_residual, constant_matrix = true, kwargs...)
    SC2 = SolverConfiguration(PDT; init = sol, maxiterations = 1, target_residual = target_residual, kwargs...)
    sol, nits = iterate_until_stationarity([SC1, SC2]; energy_integrator = EnergyIntegrator, maxsteps = maxsteps, init = sol, kwargs...)

    ## calculate mass
    Mend = sum(evaluate(MassIntegrator, sol))
    @info "M exact/start/end/difference = $(M_exact)/$M/$Mend/$(M-Mend)"

    ## calculate error
    error = evaluate(ErrorIntegratorExact, sol)
    Results[lvl, 1] = sqrt(sum(view(error, 1, :)) + sum(view(error, 2, :))) # u = (u_1,u_2)
    Results[lvl, 2] = sqrt(sum(view(error, 3, :)) + sum(view(error, 4, :)) + sum(view(error, 5, :)) + sum(view(error, 6, :))) # ∇u = (∂_x u_1,∂_y u_1, ∂_x u_2, ∂_y u_2 )
    Results[lvl, 3] = sqrt(sum(view(error, 7, :))) # ρ
    Results[lvl, 4] = sqrt(sum(view(error, 8, :)) + sum(view(error, 9, :))) # (ρ u_1 - ρ u_2)
    Results[lvl, 5] = nits

    ## print results
    print_convergencehistory(NDofs[1:lvl], Results[1:lvl, :]; X_to_h = X -> X .^ (-1 / 2), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||", "|| ϱ - ϱ_h ||", "|| ϱu - ϱu_h ||", "#its"], xlabel = "ndof")
end


## plot
plt = GridVisualizer(; Plotter = Plotter, layout = (2, 2), clear = true, size = (1000, 1000))
scalarplot!(plt[1, 1], xgrid, view(nodevalues(sol[u]; abs = true), 1, :), levels = 0, colorbarticks = 7)
vectorplot!(plt[1, 1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)), rasterpoints = 10, clear = false, title = "u_h (abs + quiver)")
scalarplot!(plt[2, 1], xgrid, view(nodevalues(sol[ϱ]), 1, :), levels = 11, title = "ϱ_h")
plot_convergencehistory!(plt[1, 2], NDofs, Results[:, 1:4]; add_h_powers = [order, order + 1], X_to_h = X -> 0.2 * X .^ (-1 / 2), legend = :best, ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||", "|| ϱ - ϱ_h ||", "|| ϱu - ϱu_h ||", "#its"])
#plot_convergencehistory!(plt[1, 1], NDofs, Results[:, 1:4]; add_h_powers = [order, order + 1], X_to_h = X -> 0.2 * X .^ (-1 / 2), legend = :best, ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||", "|| ϱ - ϱ_h ||", "|| ϱu - ϱu_h ||", "#its"])
gridplot!(plt[2, 2], xgrid)

Plotter.savefig("RigidBR_ConvParam$(conv_parameter)_p_f=$(pressure_in_f)_l_rhs=$(laplacian_in_rhs)_μ=$(μ)_cM=$(c)_reconstruct=$(reconstruct)_velocity=$(velocitytype)_ϱ=$(densitytype)_eos=$(eostype).png")




return Results, plt
end