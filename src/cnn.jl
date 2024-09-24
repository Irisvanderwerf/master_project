using Lux

# CNN construction to learn the velocity field 
function build_NN()
    return @compact(
        # Define the convolutional layers as a chain
        conv_layers = Chain(
            Conv((3, 3), 2 => 16, leakyrelu, pad=(1,1)),    # Input channels: 2, Output channels: 16
            BatchNorm(16),
            MaxPool((2, 2)),                                # Downsample to 14x14
            Conv((3, 3), 16 => 32, leakyrelu, pad=(1,1)),   # Keep 14x14
            BatchNorm(32),
            Conv((3, 3), 32 => 64, leakyrelu, pad=(1,1)),   # Keep 14x14
            BatchNorm(64),
            Upsample((2, 2)),                               # Upsample back to 28x28
            Conv((3, 3), 64 => 32, leakyrelu, pad=(1,1)),   # Keep 28x28
            BatchNorm(32),
            Conv((3, 3), 32 => 16, leakyrelu, pad=(1,1)),   # Keep 28x28
            BatchNorm(16),
            Conv((3, 3), 16 => 1, pad=1, use_bias=false),   # Keep 28x28, final velocity field
        )
    ) do x
        # Extract the input image and time
        I_sample, t_sample = x
        # Reshape t_sample to match the spatial dimensions of I_sample (28, 28, 1, B)
        t_sample_reshaped = repeat(t_sample, 28, 28, 1, 1)
        # Concatenate the time t along the channel dimension
        x = cat(I_sample, t_sample_reshaped, dims=3)
        # Pass through the convolutional layers
        @return conv_layers(x)
    end
end