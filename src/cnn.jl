using Lux


function sinusoidal_embedding(x, min_freq::AbstractFloat, max_freq::AbstractFloat, embedding_dims::Int, dev=gpu_device())
    if length(size(x)) != 4
        x = reshape(x, (1, 1, 1, size(x)[end]))
        # throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    # define frequencies
    # LinRange requires @adjoint when used with Zygote
    # Instead we manually implement range.
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> dev
    # @assert length(freqs) == div(embedding_dims, 2)
    # @assert size(freqs) == (div(embedding_dims, 2),)

    angular_speeds = reshape(2.0f0 * π * freqs, (1, 1, length(freqs), 1))
    # @assert size(angular_speeds) == (1, 1, div(embedding_dims, 2), 1)

    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)
    # @assert size(embeddings) == (1, 1, embedding_dims, size(x, 4))

    return embeddings#dropdims(embeddings, dims=(1, 2)) #embeddings
end


# CNN construction to learn the velocity field 
function build_NN()
    return @compact(
        # Define the convolutional layers as a chain
        conv_in = Conv((3, 3), 1 => 32, leakyrelu, pad=(1,1)),
        conv_layers = Chain(
            Conv((3, 3), 64 => 64, leakyrelu, pad=(1,1)),    # Input channels: 2, Output channels: 16
            # BatchNorm(64),
            # MaxPool((2, 2)),                                # Downsample to 14x14
            # Conv((3, 3), 32 => 32, leakyrelu, pad=(1,1)),   # Keep 14x14
            # # BatchNorm(64),
            # Conv((3, 3), 32 => 32, leakyrelu, pad=(1,1)),   # Keep 14x14
            # BatchNorm(64),
            # Upsample((2, 2)),                               # Upsample back to 28x28
            Conv((3, 3), 64 => 64, leakyrelu, pad=(1,1)),   # Keep 28x28
            # BatchNorm(64),
            Conv((3, 3), 64 => 64, leakyrelu, pad=(1,1)),   # Keep 28x28
            # BatchNorm(64),
            Conv((3, 3), 64 => 1, pad=1, use_bias=false),   # Keep 28x28, final velocity field
        ),
        t_embedding = t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, 32)
    ) do x
        # Extract the input image and time
        I_sample, t_sample = x

        x = conv_in(I_sample)

        # Reshape t_sample to match the spatial dimensions of I_sample (28, 28, 1, B)
        t_sample_reshaped = repeat(t_sample, 28, 28, 1, 1)
        t_sample_reshaped = t_embedding(t_sample_reshaped)

        # Concatenate the time t along the channel dimension
        x = cat(x, t_sample_reshaped, dims=3)
        # Pass through the convolutional layers
        @return conv_layers(x)
    end
end