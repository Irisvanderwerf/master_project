using Lux
using Random

# Define the sinusoidal embedding function 
function sinusoidal_embedding(x, min_freq::AbstractFloat, max_freq::AbstractFloat, embedding_dims::Int)# , dev=gpu_device())
    # Determine range of frequencies between min_freq and max_freq. 
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) # |> dev

    angular_speeds = reshape(2.0f0 * π * freqs, (1, 1, length(freqs), 1))

    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)
    return embeddings
end
# This function transforms a time variable into a higher-dimensional space using sine and cosine functions at different frequencies. 
# This will help the model to learn the temporal dependencies. 

# Define the U-Net building block with Conv and ReLU activation followed by batch normalization
function ConvBlock(in_channels, out_channels)
    return Chain(
        Conv((3, 3), in_channels => out_channels, leakyrelu, pad=(1,1)),
        Conv((3, 3), out_channels => out_channels, leakyrelu, pad=(1,1)),
        MaxPool((2, 2))
    )
end

# Define the up-sampling block
function UpBlock(in_channels, out_channels)
    return Chain(
        Upsample((2, 2)),  # Upsampling to double the spatial dimensions
        Conv((3, 3), in_channels => out_channels, leakyrelu, pad=(1, 1)),
        Conv((3, 3), out_channels => out_channels, leakyrelu, pad=(1, 1)),
    )
end

# Define the U-Net architecture
function UNet()
    return @compact(
        # We start with 64 input channels
        # Down-sampling path
        down1 = ConvBlock(64, 128),  # Start with 2 input channels for I_sample and t_sample
        down2 = ConvBlock(128, 256),
        down3 = ConvBlock(256, 512),
        
        bottom = Chain(
            Conv((3, 3), 512 => 1024, leakyrelu, pad=(1, 1)),
            Conv((3, 3), 1024 => 1024, leakyrelu, pad=(1, 1)),
        ),

        # Up-sampling path - channel reduction 
        up3 = UpBlock(1536, 512),   
        up2 = UpBlock(768, 256),
        up1 = UpBlock(384, 128),
        
        final_conv = Conv((1, 1), 128 => 1, pad=(2,2))  # Output layer with one channel
    ) do x
        x_down1 = down1(x)
        println("after down1 shape: ", size(x_down1))
        x_down2 = down2(x_down1)
        println("after down2 shape: ", size(x_down2))
        x_down3 = down3(x_down2)
        println("after down3 shape: ", size(x_down3))
    
        x = bottom(x_down3)  
        println("After bottom shape: ", size(x))
        
        # Up-sample with skip connections
        x = cat(x, x_down3, dims=3)
        println("After concatenating shape: ", size(x))
        x = up3(x)
        println("After up3 shape: ", size(x))

        target_size = size(x,1)
        crop_start = 1  # Starting row index
        crop_end = target_size  # Ending row index (6 in this case)
        x = cat(x, view(x_down2, crop_start:crop_end, crop_start:crop_end, :, :), dims=3)
        println("After concatenating shape: ", size(x))
        x = up2(x)
        println("After up2 shape: ", size(x))

        target_size = size(x,1)
        crop_start = 1  # Starting row index
        crop_end = target_size  # Ending row index (6 in this case)
        x = cat(x, view(x_down1, crop_start:crop_end, crop_start:crop_end, :, :), dims=3)
        println("After concatenating shape: ", size(x))
        x = up1(x)
        println("After up1 shape: ", size(x))
        
        @return final_conv(x)  # Final 1x1 convolution to reduce channels - Output shape: (28, 28, 1, 32)
    end
end

# Define the full U-Net with time embedding
function build_full_unet()
    return @compact(
        conv_in = Conv((3, 3), 1 => 32, leakyrelu, pad=(1,1)),
        u_net = UNet(),
        t_embedding = t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, 32)
    ) do x
        # Extract the input image and time
        I_sample, t_sample = x

        x = conv_in(I_sample) # shape: (28, 28, 32, 32)

        # Reshape t_sample to match the spatial dimensions of I_sample (28, 28, 1, B)
        t_sample_reshaped = repeat(t_sample, 28, 28, 1, 1)
        t_sample_reshaped = t_embedding(t_sample_reshaped) # shape: (28, 28, 32, 32)

        # Concatenate the time t along the channel dimension
        x = cat(x, t_sample_reshaped, dims=3) # shape: (28, 28, 64, 32)
        # Pass through the convolutional layers
        return u_net(x)
    end
end

# using ComponentArrays

# # Define input parameters
# batch_size = 32
# img_size = (28, 28)  # Size of MNIST images
# # Create a random input image tensor of shape (28, 28, 1, batch_size)
# I_sample = rand(Float32, img_size[1], img_size[2], 1, batch_size)
# # Create a random time sample tensor of shape (1, 1, 1, batch_size)
# t_sample = rand(Float32, 1, 1, 1, batch_size)

# velocity_cnn = build_full_unet()

# ### Initialize network parameters and state ###
# ps, st = Lux.setup(Random.default_rng(), velocity_cnn)
# ps = ComponentArray(ps)
# println("The number of parameters of the CNN: ", length(ps))

# # Forward pass through the network with example inputs
# output, st = velocity_cnn((I_sample, t_sample), ps, st)
# # Print the output shape to verify
# println("Output shape: ", size(output))  