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

# Define zeros. 
z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)
ArrayType = CUDA.functional() ? CuArray : Array
CUDA.allowscalar(false)

function Q(u, params)
    (; K, Kf, k) = params
    n = size(u, 1)
    Kz = K - Kf # The highest frequencies of u are cut-off prevent aliasing. 

    # Remove aliasing components by zero-padding: pads the central part of the array with zerios, preserving the low-frequency components and discarding the high-frequency components.
    # z(a,b,c) generates a zero array of size axbxc. 
    uf = [
        u[1:Kf, 1:Kf, :] z(Kf, 2Kz, 2) u[1:Kf, end-Kf+1:end, :]
        z(2Kz, Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, Kf, 2)
        u[end-Kf+1:end, 1:Kf, :] z(Kf, 2Kz, 2) u[end-Kf+1:end, end-Kf+1:end, :]
    ]

    # Spatial velocity
    v = real.(ifft(uf, (1, 2))) # ifft: inverse fourier transform - converts the modified spectral data uf back to physical space (Applied on the first two dimensions)
    vx, vy = eachslice(v; dims = 3)

    # Quadractic terms in space
    vxx = vx .* vx
    vxy = vx .* vy
    vyy = vy .* vy
    v2 = cat(vxx, vxy, vxy, vyy; dims = 3) 
    v2 = reshape(v2, n, n, 2, 2) 
    # Concatenated and reshaped into a 4-dimensional array to represent the outer product of the velocity components. 

    # Quadractic terms in spectral space
    q = fft(v2, (1, 2)) # fft: fourier transform. 
    qx, qy = eachslice(q; dims = 4)

    # Compute partial derivatives in spectral space
    ∂x = 2f0π * im * k
    ∂y = 2f0π * im * reshape(k, 1, :)
    q = @. -∂x * qx - ∂y * qy

    # Zero out high wave-numbers (is this necessary?)
    q = [
        q[1:Kf, 1:Kf, :] z(Kf, 2Kz, 2) q[1:Kf, Kf+2Kz+1:end, :]
        z(2Kz, Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, Kf, 2)
        q[Kf+2Kz+1:end, 1:Kf, :] z(Kf, 2Kz, 2) q[Kf+2Kz+1:end, Kf+2Kz+1:end, :]
    ]

    q
end

# F computes the unprojected momentum right hand side ̂F. It also includes the closure term (if any). 
function F(u, params)
    (; normk, nu, f, m, θ) = params # m the possible closure term. 
    q = Q(u, params)
    du = @. q - nu * (2f0π)^2 * normk * u + f
    # If m is provided, this is added to du. 
    isnothing(m) || (du += m(u, θ))
    du
end

# The projector P uses pre-assembled matrices. The resulting array contains the transformed (projected) x and y components.
function project(u, params)
    (; Pxx, Pxy, Pyy) = params
    ux, uy = eachslice(u; dims = 3)
    dux = @. Pxx * ux + Pxy * uy
    duy = @. Pxy * ux + Pyy * uy
    cat(dux, duy; dims = 3)
end

# The following function performsone RK4 time step. 
function step_rk4(u0, params, dt)
    a = (
        (0.5f0,),
        (0.0f0, 0.5f0),
        (0.0f0, 0.0f0, 1.0f0),
        (1.0f0 / 6.0f0, 2.0f0 / 6.0f0, 2.0f0 / 6.0f0, 1.0f0 / 6.0f0),
    )
    u = u0
    k = ()
    for i = 1:length(a)
        ki = project(F(u, params), params)
        k = (k..., ki)
        u = u0
        for j = 1:i
            u += dt * a[i][j] * k[j]
        end
    end
    u
end

# For plotting, the spatial vorticity can be useful. 
function vorticity(u, params)
    (; k) = params
    ∂x = 2f0π * im * k
    ∂y = 2f0π * im * reshape(k, 1, :)
    ux, uy = eachslice(u; dims = 3)
    ω = @. -∂y * ux + ∂x * uy
    real.(ifft(ω))
end

# This function creates a random Gaussian force field.
function gaussian(x; σ = 0.1f0)
    n = length(x)
    xf, yf = rand(), rand() # random values between 0 and 1 -> determine the center of the Gaussian function. 
    f = [
        exp(-(x - xf + a)^2 / σ^2 - (y - yf + a)^2 / σ^2) for x ∈ x, y ∈ x,
        a in (-1, 0, 1), b in (-1, 0, 1) 
    ] ## periodic padding: shifting the center in both the x and y directions, ensuring the Gaussian field wraps around periodically. 
    f = reshape(sum(f; dims = (3, 4)), n, n) # sums the Gaussian fields across the third and fourth dimensions, which corresponds to the shifts a and b. 
    f = exp(im * rand() * 2.0f0π) * f ## Rotate f in the complex plane.
    cat(real(f), imag(f); dims = 3)
end

# For the initial conditions, we create a random spectrum with some decay. 
# Note that the initial conditions are projected onto the divergence free space at the end.
function create_spectrum(params; A, σ, s)
    (; x, k, K) = params
    T = eltype(x) # element type of x. 
    kx = k
    ky = reshape(k, 1, :) # reshapes k to create a 2D wavenumber grid.
    τ = 2.0f0π
    a = @. A / sqrt(τ^2 * 2σ^2) *
       exp(-(kx - s)^2 / 2σ^2 - (ky - s)^2 / 2σ^2 - im * τ * rand(T))
    # Gaussian function controls the distribution of energy in the spectral space. 
    # The random phase term adds randomness to the spectrum. 
    a
end
# Returns a spectrum which is a complex 2D arrayrepresenting the Fourier coefficientsfor one component of the velocity field.

function random_field(params; A = 1.0f6, σ = 30.0f0, s = 5.0f0)
    ux = create_spectrum(params; A, σ, s)
    uy = create_spectrum(params; A, σ, s)
    u = cat(ux, uy; dims = 3)
    u = real.(ifft(u, (1, 2))) # physical space.
    u = fft(u, (1, 2)) # spectral space.
    project(u, params) # ensure that the velocity field has the desired properties, such as divergence-free.
end

# store parameters and precomputed operators in a named tuple to toss around. 
# Useful when  we work with multiple resolutions. 
function create_params(
    K;
    nu,
    f = z(2K, 2K),
    m = nothing,
    θ = nothing,
    anti_alias_factor = 2 / 3,
)
    Kf = round(Int, anti_alias_factor * K) # effective grid resolution after applying the anti-aliasing factor. 
    N = 2K # total grid size.
    x = LinRange(0.0f0, 1.0f0, N + 1)[2:end] # spatial coordinates of the grid.

    # Vector of wavenumbers
    k = ArrayType(fftfreq(N, Float32(N))) # wave number array: array of frequenciescorresponding to the Fourier modes of the grid. 
    normk = k .^ 2 .+ k' .^ 2 # 

    # Projection components
    kx = k
    ky = reshape(k, 1, :)
    Pxx = @. 1 - kx * kx / (kx^2 + ky^2)
    Pxy = @. 0 - kx * ky / (kx^2 + ky^2) 
    Pyy = @. 1 - ky * ky / (kx^2 + ky^2)

    # The zeroth component is currently `0/0 = NaN`. For `CuArray`s,
    # we need to explicitly allow scalar indexing.

    CUDA.@allowscalar Pxx[1, 1] = 1 # the x-projecction at the zero frequency is complete.
    CUDA.@allowscalar Pxy[1, 1] = 0 # there's no cross-term at zero frequency.
    CUDA.@allowscalar Pyy[1, 1] = 1 # the y-projection is complete at zero frequency. 

    # Closure model
    m = nothing
    θ = nothing

    (; x, N, K, Kf, k, nu, normk, f, Pxx, Pxy, Pyy, m, θ)
end