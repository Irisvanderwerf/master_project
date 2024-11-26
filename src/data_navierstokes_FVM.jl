using FFTW
using LinearAlgebra
using Random
using CUDA
using ComponentArrays
using IJulia
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Plots
using Random
using Zygote
using KernelAbstractions 

using IncompressibleNavierStokes
const INS = IncompressibleNavierStokes
using OrdinaryDiffEq

# The following function performsone RK4 time step. 
function step_rk4(u0, dt, F)
    a = (
        (0.5f0,),
        (0.0f0, 0.5f0),
        (0.0f0, 0.0f0, 1.0f0),
        (1.0f0 / 6.0f0, 2.0f0 / 6.0f0, 2.0f0 / 6.0f0, 1.0f0 / 6.0f0),
    )
    u = u0
    k = ()
    for i = 1:length(a)
        ki = F(u, nothing, 0.0)
        k = (k..., ki)
        u = u0
        for j = 1:i
            @.  u += dt * a[i][j] * k[j]
        end
    end
    u
end

# The face filtering averaging function for 2D NS-Equations. 
function face_averaging_velocity_2D(u_DNS, N_DNS, N_LES) 
    # Calculate the DNS cells per LES cell
    dx = N_DNS ÷ N_LES
    dy = N_DNS ÷ N_LES

    # Initialize the LES face-averaged velocity array
    u_LES = zeros(N_LES, N_LES, 2)

    # Loop over LES cells and average DNS values at corresponding faces
    for i in 1:N_LES
        for j in 1:N_LES
            for comp in 1:2  # Loop over velocity components (x and y)
                x_start = max(1, (i-1)*dx + 1)
                x_end   = min(N_DNS, i*dx)
                y_start = max(1, (j-1)*dy + 1)
                y_end   = min(N_DNS, j*dy)

                subdomain = u_DNS[x_start:x_end, y_start:y_end, comp]
                u_LES[i, j, comp] = mean(subdomain)
            end
        end
    end

    return u_LES  
end

face_average_syver(u, setup_les, comp) = face_average_syver!(INS.vectorfield(setup_les), u, setup_les, comp)


function face_average_syver!(v, u, setup_les, comp)
    (; grid, backend, workgroupsize) = setup_les
    (; dimension, Nu, Iu) = grid
    D = dimension()
    @kernel function Φ!(v, u, ::Val{α}, face, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v))
        for i in face
            s += u[J+i, α]
        end
        v[I0+I, α] = s / comp^(D - 1)
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = getoffset(Iu[α])
        face = CartesianIndices(ntuple(β -> β == α ? (comp:comp) : (1:comp), D))
        Φ!(backend, workgroupsize)(v, u, Val(α), face, I0; ndrange)
    end
    v
end

# Function to compute mean and standard deviation per channel
function compute_mean_std(training_set)
    means = [mean(training_set[:,:,c,:]) for c in 1:2]
    stds = [std(training_set[:,:,c,:]) for c in 1:2]
    return means, stds
end

# Function to standardize the training set per channel
function standardize_training_set_per_channel(training_set, means, stds)
    standardized_set = similar(training_set)
    for c in 1:2
        standardized_set[:,:,c,:] .= (training_set[:,:,c,:] .- means[c]) ./ stds[c]
    end
    return standardized_set
end

# Function of the inverse standardization per channel
function inverse_standardize_set_per_channel(training_set, means, stds)
    inverse_standardized_set = similar(training_set)
    for c in 1:2
        inverse_standardized_set[:,:,c,:] .= (training_set[:,:,c,:] .* stds[c]) .+ means[c]
    end
    return inverse_standardized_set
end