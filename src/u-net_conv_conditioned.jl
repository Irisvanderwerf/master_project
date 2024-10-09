using Lux
using Random
using NNlib
using LuxCUDA

# Sinusoidal embedding for the time
function sinusoidal_embedding(x, min_freq::AbstractFloat, max_freq::AbstractFloat, embedding_dims::Int)
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) # |> gpu_device()
    
    angular_speeds = reshape(2.0f0 * π * freqs, (1, 1, length(freqs), 1))
    
    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)

    return dropdims(embeddings, dims=(1, 2))
end

# ConvNextBlock definition where the time embedding perturbs the convolution
function ConvNextBlock(;
    in_channels::Int, 
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1
)
    @compact(
        ds_conv = Conv((7, 7), in_channels => in_channels; pad=3),  
        pars_mlp = Lux.Dense(embedding_dim => in_channels),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            Conv((3, 3), in_channels => in_channels * multiplier, pad=(1,1)),  
            NNlib.gelu,
            InstanceNorm(in_channels * multiplier),
            Lux.Conv((3, 3), in_channels * multiplier => out_channels, pad=(1,1))  
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)  
    ) do x
        x, pars = x
        
        h = ds_conv(x) 
    
        # Process time embeddings
        pars = pars_mlp(pars) 
        pars = reshape(pars, 1, 1, size(pars)...) 
    
        # Add time embedding to conv features
        h = h .+ pars 
    
        h = conv_net(h) 
        
        # Add residual connection
        @return h .+ res_conv(x) 
    end
end

# Down-sampling block with ConvNextBlock
function DownBlock(in_channels, out_channels, embedding_dim)
    return @compact(
        convnext_block = ConvNextBlock(in_channels=in_channels, out_channels=out_channels, embedding_dim=embedding_dim),
        down_conv = Conv((4, 4), out_channels => out_channels; pad=1, stride=(2, 2)),
    ) do x
        x, pars = x
        x = convnext_block((x, pars)) 
        @return down_conv(x)
    end
end

# Up-sampling block with ConvNextBlock
function UpBlock(in_channels, out_channels, embedding_dim)
    return @compact(
        convnext_block = ConvNextBlock(in_channels=in_channels, out_channels=out_channels, embedding_dim=embedding_dim),
        up_conv = ConvTranspose((4, 4), out_channels => out_channels; pad=1, stride=(2, 2)),
    ) do x
        x, pars = x
        x = convnext_block((x, pars))  
        @return up_conv(x)
    end
end

# Main U-Net architecture with time embedding and ConvNextBlocks
function UNet(
    in_channels = 1,
    out_channels = 1,
    hidden_channels = [16, 32, 64],
    embedding_dim = 8,
)
    return @compact(
        # Down-sampling path
        down1 = DownBlock(in_channels, hidden_channels[1], embedding_dim),
        down2 = DownBlock(hidden_channels[1], hidden_channels[2], embedding_dim),
        down3 = DownBlock(hidden_channels[2], hidden_channels[3], embedding_dim),

        bottom = ConvNextBlock(in_channels=hidden_channels[3], out_channels=hidden_channels[3], embedding_dim=embedding_dim),
        
        # bottom = Chain(
        #     Conv((3, 3), hidden_channels[3] => 2 * hidden_channels[3], leakyrelu; pad=1),
        #     Conv((3, 3), 2 * hidden_channels[3] => hidden_channels[3], leakyrelu; pad=1),
        # ),

        # Up-sampling path
        up3 = UpBlock(2 * hidden_channels[3], hidden_channels[2], embedding_dim ),   
        up2 = UpBlock(2 * hidden_channels[2], hidden_channels[1], embedding_dim),
        up1 = UpBlock(2 * hidden_channels[1], hidden_channels[1], embedding_dim),
        
        # Output layer
        final_conv = Conv((1, 1), hidden_channels[1] => out_channels, use_bias=false)

    ) do x
        x, pars = x 

        # Down-sampling path
        x_down1 = down1((x, pars)) # size: (16,16,hidden_channels[1],batch_size)
        x_down2 = down2((x_down1, pars)) # size: (8,8,hidden_channels[2],batch_size)
        x_down3 = down3((x_down2, pars)) # size: (4,4,hidden_channels[3],batch_size)

        x = bottom((x_down3, pars)) # The size of bottom: (4,4,hidden_channels[3],32)

        # Up-sampling with skip connections
        x = cat(x, x_down3, dims=3) # (4,4,2*hidden_channels[3],batch_size)
        x = up3((x, pars)) # The size after first up: (8,8,hidden_channels[2],batch_size)
        
        x = cat(x, x_down2, dims=3) # (8,8,2*hidden_channels[2],batch_size)
        x = up2((x, pars)) # The size after second up: (16,16,hidden_channels[1], batch_size)
        
        x = cat(x, x_down1, dims=3) # (16,16,2*hidden_channels[1],batch_size)
        x = up1((x, pars)) # The size after last up: (32,32,hidden_channels[1],batch_size)

        @return final_conv(x) # (32,32,1,batch_size)
    end
end

# Full U-Net model with time embedding
function build_full_unet(embedding_dim = 8, hidden_channels = [16, 32, 64], t_pars_embedding_dim = 8)
    return @compact(
        conv_in = Conv((3, 3), 1 => (embedding_dim/2), leakyrelu, pad=(1,1)),
        u_net = UNet(embedding_dim, 1, hidden_channels, embedding_dim), 
        t_embedding = Chain(
            t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, t_pars_embedding_dim),
            Lux.Dense(t_pars_embedding_dim => embedding_dim),
            NNlib.gelu,
            Lux.Dense(embedding_dim => embedding_dim),
            NNlib.gelu,
          )
    ) do x
        I_sample, t_sample, cond = x # size I_sample: (32,32,1,batch_size), t_sample: (1,1,1,batch_size), cond: (32,32,1,batch_size)
        x = conv_in(I_sample) # size: (32, 32, embedding_dim/2, batch_size)
        cond_in = conv_in(cond) # size: (32, 32, embedding_dim/2, batch_size)
        x = cat(x, cond_in, dims=3) # size: (32,32,embedding_dim, batch_size)

        t_sample_embedded = t_embedding(t_sample) # size: (embedding_dim, batch_size)

        @return u_net((x, t_sample_embedded))
    end
end