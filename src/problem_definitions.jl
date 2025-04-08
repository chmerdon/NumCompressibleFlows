abstract type TestDensity end
abstract type LinearDensity <: TestDensity end
abstract type ExponentialDensity <: TestDensity end
abstract type ExponentialDensityRBR <: TestDensity end

abstract type TestVelocity end
abstract type ZeroVelocity <: TestVelocity end
abstract type ConstantVelocity <: TestVelocity end
abstract type P7VortexVelocity <: TestVelocity end
abstract type RigidBodyRotation <: TestVelocity end

abstract type EOSType end
abstract type IdealGasLaw <: EOSType end
abstract type PowerLaw{γ} <: EOSType end


abstract type GridFamily end
abstract type Mountain2D <: GridFamily end
abstract type UnitSquare <: GridFamily end
abstract type UniformUnitSquare <: UnitSquare end # no grid function yet
abstract type UnstructuredUnitSquare <: UnitSquare end

function grid(::Type{<:Mountain2D}; nref = 1, kwargs...)
    grid_builder = (nref) -> SimplexGridFactory.simplexgrid(
            Triangulate;
            points = [0 0; 0.2 0; 0.3 0.2; 0.45 0.05; 0.55 0.35; 0.65 0.2; 0.7 0.3; 0.8 0; 1 0; 1 1 ; 0 1]',
            bfaces = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 11; 11 1]',
            bfaceregions = ones(Int, 11),
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref) / 2]
        )
    return grid_builder(nref)
end

function grid(::Type{<:UnstructuredUnitSquare}; nref = 1, kwargs...)
    grid_builder = (nref) -> SimplexGridFactory.simplexgrid(
            Triangulate;
            points = [0 0; 1.0 0; 1.0 1.0; 0.0 1.0]',
            bfaces = [1 2; 2 3; 3 4; 4 1]',
            bfaceregions = [1,2,3,4],
            regionpoints = [0.5 0.5;]',
            regionnumbers = [1],
            regionvolumes = [4.0^-(nref) / 2]
        )
    return grid_builder(nref)
end

function grid(::Type{<:UniformUnitSquare}; nref = 1, kwargs...)
    grid_builder = (nref) -> uniform_refine(simplexgrid(0:0.5:1,0:0.5:1), nref)
    return grid_builder(nref)
end

streamfunction(::Type{<:ZeroVelocity};ufac = 1, kwargs...) = 0* ufac *x
streamfunction(::Type{<:ConstantVelocity};ufac = 1, kwargs...) = - ufac * y
streamfunction(::Type{<:P7VortexVelocity};ufac = 1, kwargs...) = ufac * x^2 * y^2 * (x - 1)^2 * (y - 1)^2
streamfunction(::Type{<:RigidBodyRotation};ufac = 1, kwargs...) = (ufac/2) * (x^2 + y^2)

already_divfree(::Type{<:TestVelocity}) = false
already_divfree(::Type{<:RigidBodyRotation}) = true

inflow_regions(::Type{<:ZeroVelocity}, gridtype) = []
inflow_regions(::Type{<:ConstantVelocity}, gridtype) = [1,3,4]
inflow_regions(::Type{<:P7VortexVelocity}, gridtype) = []
inflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitSquare}) = [1,2]
outflow_regions(::Type{<:ZeroVelocity}, gridtype) = []
outflow_regions(::Type{<:ConstantVelocity}, gridtype) = [2]
outflow_regions(::Type{<:P7VortexVelocity}, gridtype) = []
outflow_regions(::Type{<:RigidBodyRotation}, ::Type{<:UnitSquare}) = [3,4]


#density(::Type{<:ExponentialDensity}; c = 1, M = 1, kwargs...) = exp(- y^3 / 3 * c) / M 
density(::Type{<:ExponentialDensityRBR}; c = 1, M = 1, kwargs...) = M * exp( (x^2+y^2) /  (2*c))
density(::Type{<:ExponentialDensity}; c = 1, M = 1, kwargs...) = exp(- y /  c) / M  # e^(-y/c_M)/ M  ... Objection: whrere is x^3 ?
density(::Type{<:LinearDensity}; c = 1, M = 1, kwargs...) = ( 1+(x-(1/2))/c )/M 


function prepare_data(TVT::Type{<:TestVelocity}, TDT::Type{<:TestDensity}, EOSType::Type{<:EOSType}; M = 1, c = 1, μ = 1, ufac = 1, laplacian_in_rhs = true, pressure_in_f = false , λ = -2*μ / 3,conv_parameter = 0, kwargs...)

    ## get stream function and density for test types
    ξ = streamfunction(TVT; ufac = ufac, kwargs...)
    ϱ = density(TDT; c = c, M = M, kwargs...)
    
    ## gradient of stream function
    ∇ξ = Symbolics.gradient(ξ, [x, y])

    ## velocity u = curl ξ / ϱ
    if already_divfree(TVT)
        u = [-∇ξ[2], ∇ξ[1]]
    else 
        u = [-∇ξ[2], ∇ξ[1]] ./ ϱ
    end

    ## gradient of velocity
    ∇u = Symbolics.jacobian(u, [x, y])
    ∇u_reshaped = [∇u[1, 1], ∇u[1, 2], ∇u[2, 1], ∇u[2, 2]] # [∇u1,∇u2]

    ## Laplacian
    Δu = [
        (Symbolics.gradient(∇u[1, 1], [x]) + Symbolics.gradient(∇u[1, 2], [y]))[1],
        (Symbolics.gradient(∇u[2, 1], [x]) + Symbolics.gradient(∇u[2, 2], [y]))[1],
    ]
    ## for the λ∇(∇⋅u) term 
    ∇divu = [
        (Symbolics.gradient(∇u[1, 1], [x]) + Symbolics.gradient(∇u[2, 2], [x]))[1],
        (Symbolics.gradient(∇u[1, 1], [y]) + Symbolics.gradient(∇u[2, 2], [y]))[1],
    ]

    ## for the convection term ϱ(u ⋅∇)u

    conv = conv_parameter * [ u[1] * ∇u[1, 1] + u[2] * ∇u[1, 2],
     u[1] * ∇u[2, 1] +  u[2] * ∇u[2, 2] ] .* ϱ
    @show conv 

    # L(u) + ∇p = f + ϱg with L(u) = -μ Δu - λ ∇(∇⋅u) + ϱ(u.∇)u 
    if pressure_in_f # Gradient_robustness 
        if EOSType <: IdealGasLaw
            f = c * Symbolics.gradient(ϱ, [x, y]) # f = ∇p 
        elseif EOSType <: PowerLaw
            γ = EOSType.parameters[1]
            @assert γ > 1
            f =  c * Symbolics.gradient(ϱ^γ, [x, y]) # f = ∇p 
        end
        if laplacian_in_rhs
            g =  0 * Δu 
            f += - μ * Δu - λ*∇divu + conv   # f = L(u) + ∇p (everything in f)
        else
            g = - μ * Δu / ϱ - λ*∇divu / ϱ + conv /ϱ  # ϱg = L(u)
        end
           
    else # Well_balancedness 
        if EOSType <: IdealGasLaw
            g = c * Symbolics.gradient(log(ϱ), [x, y]) # ϱg = ∇p  
        elseif EOSType <: PowerLaw
            γ = EOSType.parameters[1]
            @assert γ > 1
            g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, [x, y]) # ϱg = ∇p 
        end
        if laplacian_in_rhs 
            f =  0 * Δu 
            g += - μ * Δu / ϱ  - λ*∇divu / ϱ + conv /ϱ  # ϱg = L(u) + ∇p (everything in g)
            
        else
            f = - μ * Δu  - λ*∇divu + conv  # f = L(u)
        end
    end
    @info f, g


## Christian's def of f & g 
   #=
    if EOSType <: IdealGasLaw
        g = c * Symbolics.gradient(log(ϱ), [x, y])
    elseif EOSType <: PowerLaw
        γ = EOSType.parameters[1]
        @assert γ > 1
        g =  c* γ*ϱ^(γ-2) * Symbolics.gradient(ϱ, [x, y]) # testing welll-balance
    end

    ## gravity ϱg = - Δu + ϱ∇log(ϱ)
    if laplacian_in_rhs 
        f = - μ * Δu  - λ*∇divu   
        
    else
        g -= - μ * Δu / ϱ  - λ*∇divu / ϱ
        f = 0 * Δu 
    end
   =#

    #Δu = Symbolics.derivative(∇u[1,1], [x]) + Symbolics.derivative(∇u[2,2], [y])

    ϱ_eval = build_function(ϱ, x, y, expression = Val{false})
    u_eval = build_function(u, x, y, expression = Val{false})[2]
    ∇u_eval = build_function(∇u_reshaped, x, y, expression = Val{false})[2]
    g_eval = build_function(g, x, y, expression = Val{false})[2]
    f_eval = build_function(f, x, y, expression = Val{false})[2]

    ϱ!(result, qpinfo) = (result[1] = ϱ_eval(qpinfo.x[1], qpinfo.x[2]);)
    function kernel_gravity!(result, input, qpinfo)
        g_eval(result, qpinfo.x[1], qpinfo.x[2]) # qpinfo.x[1] is x and qpinfo.x[2] is y
        return result .*= input[1] # what does it mean ?
    end
    function kernel_rhs!(result, qpinfo)
        return f_eval(result, qpinfo.x[1], qpinfo.x[2])
    end

    u!(result, qpinfo) = (u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
    ∇u!(result, qpinfo) = (∇u_eval(result, qpinfo.x[1], qpinfo.x[2]);)
    return ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u!
end


function filename(data)
    # problem parameters
    μ = data["μ"]
    λ = data["λ"]
    γ = data["γ"]
    c = data["c"]
    M = data["M"]
    # solving options
    τfac = data["τfac"]
    ufac = data["ufac"]
    nrefs = data["nrefs"]
    order = data["order"]
    reconstruct = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps = data["maxsteps"]
    pressure_stab = data["pressure_stab"]
    # data of the problem
    velocitytype = data["velocitytype"]
    densitytype = data["densitytype"]
    eostype = data["eostype"]
    gridtype = data["gridtype"]
    laplacian_in_rhs = data["laplacian_in_rhs"]
    # sname
    sname = savename((@dict μ λ γ c M τfac ufac nrefs order reconstruct target_residual maxsteps pressure_stab velocitytype densitytype eostype gridtype laplacian_in_rhs))
    sname = "data/projects/compressible_stokes/" * sname
    return sname
end

function run_single(data; kwargs...)
    # problem parameters
    μ = data["μ"]
    λ = data["λ"]
    γ = data["γ"]
    c = data["c"]
    M = data["M"]
    # solving options
    τfac = data["τfac"]
    ufac = data["ufac"]
    nrefs = data["nrefs"]
    order = data["order"]
    reconstruct = data["reconstruct"]
    target_residual = data["target_residual"]
    maxsteps = data["maxsteps"]
    pressure_stab = data["pressure_stab"]
    # data of the problem
    velocitytype = data["velocitytype"]
    densitytype = data["densitytype"]
    eostype = data["eostype"]
    gridtype = data["gridtype"]
    pressure_in_f = data["pressure_in_f"]
    laplacian_in_rhs = data["laplacian_in_rhs"]
    data = Dict{String, Any}(data)
    @show data, typeof(data)

    ## load data for testcase
    ϱ!, kernel_gravity!, kernel_rhs!, u!, ∇u! = prepare_data(velocitytype, densitytype, eostype; laplacian_in_rhs = laplacian_in_rhs, pressure_in_f = pressure_in_f, M = M, c = c, μ = μ, λ = λ,γ=γ, ufac = ufac , nrefs = nrefs)
    # added new for the type version
    xgrid = NumCompressibleFlows.grid(gridtype; nref = nrefs)
    #xgrid = grid(gridtype; nref = nrefs)
    M_exact = integrate(xgrid, ON_CELLS, ϱ!, 1; quadorder = 20) 
    M = M_exact
    τ = μ / (c*order^2 * M * sqrt(τfac)) # time step for pseudo timestepping
    #τ = μ / (4*order^2 * M * sqrt(τfac)) 
    @info "M = $M, τ = $τ"

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
    ## define first sub-problem: Stokes equations to solve for velocity u
    PD = ProblemDescription("Stokes problem")
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
    assign_operator!(PD, BilinearOperator([div_u]; factor = λ, store = true, kwargs...)) # Marwa div term 
    assign_operator!(PD, LinearOperator(eos!(eostype), [div(u)], [id(ϱ)]; factor = c, kwargs...)) 
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4, kwargs...))
    if kernel_rhs! !== nothing
        assign_operator!(PD, LinearOperator(kernel_rhs!, [id_u]; factor = 1, store = true, bonus_quadorder = 3 * order, kwargs...))
    end
    assign_operator!(PD, LinearOperator(kernel_gravity!, [id_u], [id(ϱ)]; factor = 1, bonus_quadorder = 3 * order, kwargs...))

    ## FVM for continuity equation
    PDT = ProblemDescription("continuity equation")
    assign_unknown!(PDT, ϱ)
    if order > 1
        assign_operator!(PDT, BilinearOperator(kernel_continuity!, [grad(ϱ)], [id(ϱ)], [id(u)]; quadorder = 2 * order, factor = -1, kwargs...))
    end
    if pressure_stab > 0
        psf = pressure_stab #* xgrid[CellVolumes][1]
        assign_operator!(PDT, BilinearOperator(stab_kernel!, [jump(id(ϱ))], [jump(id(ϱ))], [id(u)]; entities = ON_IFACES, factor = psf, kwargs...))
    end
    assign_operator!(PDT, BilinearOperator([id(ϱ)]; quadorder = 2 * (order - 1), factor = 1 / τ, store = true, kwargs...))
    assign_operator!(PDT, LinearOperator([id(ϱ)], [id(ϱ)]; quadorder = 2 * (order - 1), factor = 1 / τ, kwargs...))
    assign_operator!(PDT, BilinearOperatorDG(kernel_upwind!, [jump(id(ϱ))], [this(id(ϱ)), other(id(ϱ))], [id(u)]; quadorder = order + 1, entities = ON_IFACES, kwargs...))
    #  [jump(id(ϱ))]is test function lambda , [this(id(ϱ)), other(id(ϱ))] is the the flux multlplied by lambda_upwind. [id(u)] is the function u that is needed 
    ## prepare error calculation
    EnergyIntegrator = ItemIntegrator(energy_kernel!, [id(u)]; resultdim = 1, quadorder = 2 * (order + 1), kwargs...)
    ErrorIntegratorExact = ItemIntegrator(exact_error!(u!, ∇u!, ϱ!), [id(u), grad(u), id(ϱ)]; resultdim = 9, quadorder = 2 * (order + 1), kwargs...)
    #NDofs = zeros(Int, nrefs)
    #Results = zeros(Float64, nrefs, 5) # it is a matrix whose rows are levels and columns are 

    sol = nothing
    #xgrid = nothing
    op_upwind = 0
    # here we run only in one level instead of for Loop over the levels
    FES = [FESpace{FETypes[j]}(xgrid) for j in 1:3] # 3 because we have dim(FETypes)=3
    sol = FEVector(FES; tags = [u, ϱ, p]) # create solution vector and tag blocks with the unknowns (u,ρ,p) that has the same order as FETypes

    ## initial guess
    fill!(sol[ϱ], M) # fill block corresbonding to unknown ρ with initial value M, in Algorithm it is M/|Ω|?? We could write it as M/|Ω| and delete area from down there
    interpolate!(sol[u], u!)
    interpolate!(sol[ϱ], ϱ!)
    #NDofs[lvl] = length(sol.entries)

    ## solve the two problems iteratively [1] >> [2] >> [1] >> [2] ...
    SC1 = SolverConfiguration(PD; init = sol, maxiterations = 1, target_residual = target_residual, constant_matrix = true, kwargs...)
    SC2 = SolverConfiguration(PDT; init = sol, maxiterations = 1, target_residual = target_residual, kwargs...)
    sol, nits = iterate_until_stationarity([SC1, SC2]; energy_integrator = EnergyIntegrator, maxsteps = maxsteps, init = sol, kwargs...)
  

    ## save data
    data["ndofs"] = length(sol.entries)
    data["solution"] = sol
    data["grid"] = xgrid

    ## calculate error
    error = evaluate(ErrorIntegratorExact, sol)
    data["Error(L2,u)"] = sqrt(sum(view(error, 1, :)) + sum(view(error, 2, :))) # u = (u_1,u_2)
    data["Error(H1,u)"] = sqrt(sum(view(error, 3, :)) + sum(view(error, 4, :)) + sum(view(error, 5, :)) + sum(view(error, 6, :))) # ∇u = (∂_x u_1,∂_y u_1, ∂_x u_2, ∂_y u_2 )
    data["Error(L2,ϱ)"] = sqrt(sum(view(error, 7, :))) # ϱ
    data["Error(L2,ϱu)"]  = sqrt(sum(view(error, 8, :)) + sum(view(error, 9, :))) # (ρ u_1 - ρ u_2)
    data["nits"] = nits


    ## save to data folder
    return data


end
