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

function ResidualBlock(in_channels, out_channels)
    return @compact(
        main_conv = Chain(
            Conv((3, 3), in_channels => out_channels, leakyrelu, pad=(1,1)),
            Lux.InstanceNorm(out_channels),
            Conv((3, 3), out_channels => out_channels, leakyrelu, pad=(1,1))
        ),
        skip_conv = Conv((1, 1), in_channels => out_channels; pad=0),
    ) do x
        x_skip = skip_conv(x)
        x = main_conv(x)
        @return  x .+ x_skip
    end
end

# Define the U-Net building block with Conv and ReLU activation followed by batch normalization
function DownBlock(in_channels, out_channels)
    return @compact(
        res_block = ResidualBlock(in_channels, out_channels),
        down_conv = Conv((4, 4), out_channels => out_channels; pad=1, stride=(2, 2)),
    ) do x
        x = res_block(x)
        @return down_conv(x)
    end
    
    # return Chain(
    #     Conv((3, 3), in_channels => out_channels, leakyrelu, pad=(1,1)),
    #     Conv((3, 3), out_channels => out_channels, leakyrelu, pad=(1,1)),
    #     MaxPool((2, 2))
    # )
end

# Define the up-sampling block
function UpBlock(in_channels, out_channels)
    return @compact(
        res_block = ResidualBlock(in_channels, out_channels),
        up_conv = ConvTranspose((4, 4), out_channels => out_channels; pad=1, stride=(2, 2)),
    ) do x
        x = res_block(x)
        @return up_conv(x)
    end
    # return Chain(
    #     Upsample((2, 2)),  # Upsampling to double the spatial dimensions
    #     Conv((3, 3), in_channels => out_channels, leakyrelu, pad=(1, 1)),
    #     Conv((3, 3), out_channels => out_channels, leakyrelu, pad=(1, 1)),
    # )
end

# Define the U-Net architecture
function UNet(
    in_channels = 1,
    out_channels = 1,
    hidden_channels = [16, 32, 64],
)
    return @compact(
        # We start with 64 input channels
        # Down-sampling path
        down1 = DownBlock(in_channels, hidden_channels[1]),  # Start with 2 input channels for I_sample and t_sample
        down2 = DownBlock(hidden_channels[1], hidden_channels[2]),
        down3 = DownBlock(hidden_channels[2], hidden_channels[3]),
        
        bottom = Chain(
            Conv((3, 3), hidden_channels[3] => 2*hidden_channels[3], leakyrelu, pad=(1, 1)),
            Conv((3, 3), 2*hidden_channels[3] => hidden_channels[3], leakyrelu, pad=(1, 1)),
        ),

        # Up-sampling path - channel reduction 
        up3 = UpBlock(2*hidden_channels[3], hidden_channels[2]),   
        up2 = UpBlock(2*hidden_channels[2], hidden_channels[1]),
        up1 = UpBlock(2*hidden_channels[1], hidden_channels[1]),
        
        # changed image to size (32,32) so we can remove the padding pad=(2,2)
        final_conv = Conv((1, 1), hidden_channels[1] => out_channels, use_bias=false)  # Output layer with one channel
    ) do x
        x_down1 = down1(x)
        # println("after down1 shape: ", size(x_down1))
        x_down2 = down2(x_down1)
        # println("after down2 shape: ", size(x_down2))
        x_down3 = down3(x_down2)
        # println("after down3 shape: ", size(x_down3))


        x = bottom(x_down3)  
        # println("After bottom shape: ", size(x))
        
        # Up-sample with skip connections
        x = cat(x, x_down3, dims=3)
        # println("After concatenating shape: ", size(x))
        x = up3(x)
        # println("After up3 shape: ", size(x))

        target_size = size(x,1)
        crop_start = 1  # Starting row index
        crop_end = target_size  # Ending row index (6 in this case)
        x = cat(x, view(x_down2, crop_start:crop_end, crop_start:crop_end, :, :), dims=3)
        # println("After concatenating shape: ", size(x))
        x = up2(x)
        # println("After up2 shape: ", size(x))

        target_size = size(x,1)
        crop_start = 1  # Starting row index
        crop_end = target_size  # Ending row index (6 in this case)
        x = cat(x, view(x_down1, crop_start:crop_end, crop_start:crop_end, :, :), dims=3)
        # println("After concatenating shape: ", size(x))
        x = up1(x)
        # println("After up1 shape: ", size(x))
        
        @return final_conv(x)  # Final 1x1 convolution to reduce channels - Output shape: (28, 28, 1, 32)
    end
end

# Define the full U-Net with time embedding
function build_full_unet(
    embedding_dim = 8,
    hidden_channels = [16, 32, 64],
)
    return @compact(
        conv_in = Conv((3, 3), 1 => embedding_dim, leakyrelu, pad=(1,1)),
        u_net = UNet(2 * embedding_dim, 1, hidden_channels),
        t_embedding = t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, embedding_dim)
    ) do x
        # Extract the input image and time
        I_sample, t_sample = x

        x = conv_in(I_sample) # shape: (28, 28, 32, 32)

        # Reshape t_sample to match the spatial dimensions of I_sample (28, 28, 1, B) - changed to (32,32,1,B)
        t_sample_reshaped = repeat(t_sample, 32, 32, 1, 1)
        t_sample_reshaped = t_embedding(t_sample_reshaped) # shape: (28, 28, 32, 32)

        # Concatenate the time t along the channel dimension
        x = cat(x, t_sample_reshaped, dims=3) # shape: (28, 28, 64, 32)
        # Pass through the convolutional layers
        @return u_net(x)
    end
end