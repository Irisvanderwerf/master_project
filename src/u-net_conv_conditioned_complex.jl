using Lux
using Random
using NNlib
using LuxCUDA
using FFTW

# Sinusoidal embedding for the time
function sinusoidal_embedding(x, min_freq::AbstractFloat, max_freq::AbstractFloat, embedding_dims::Int)
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> gpu_device()
    
    angular_speeds = reshape(2.0f0 * π * freqs, (1, 1, length(freqs), 1))
    
    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)

    return dropdims(embeddings, dims=(1, 2))
end

# ConvNextBlock definition where we include time embedding - for the down scaling. (same as the one without conditioning)
function ConvNextBlock_down(;
    in_channels::Int, 
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1,
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

# ConvNextBlock definition where we include time embedding and conditioning - for upscaling part. 
function ConvNextBlock_up(;
    in_channels::Int, 
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1,
    cond_channels = 1,
)
    @compact(
        ds_conv = Conv((7, 7), in_channels => in_channels; pad=3),  
        pars_mlp = Lux.Dense(embedding_dim => in_channels),
        cond_conv = Conv((1, 1), cond_channels => in_channels; pad=0),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            Conv((3, 3), in_channels => in_channels * multiplier, pad=(1,1)),  
            NNlib.gelu,
            InstanceNorm(in_channels * multiplier),
            Lux.Conv((3, 3), in_channels * multiplier => out_channels, pad=(1,1))  
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)  
    ) do x
        x, pars, cond = x
        
        h = ds_conv(x) 
    
        # Process time embeddings
        pars = pars_mlp(pars) 
        pars = reshape(pars, 1, 1, size(pars)...) 

        # Process 2D condition (label image) features
        cond = cond_conv(cond)
    
        # Add time embedding to conv features
        h = (h .+ pars) .* cond
    
        h = conv_net(h) 
        
        # Add residual connection
        @return h .+ res_conv(x) 
    end
end

# Main U-Net architecture with time embedding and ConvNextBlocks
function UNet(
    in_channels = 1,
    out_channels = 1,
    hidden_channels = [16, 32, 64],
    embedding_dim = 8,
    cond_channels = 1,
)
    return @compact(
        # Down-sampling path - using no conditioning for down-scaling
        conv_next_down1 = ConvNextBlock_down(in_channels=in_channels, out_channels=hidden_channels[1], embedding_dim=embedding_dim),
        down1 = Conv((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),

        conv_next_down2 = ConvNextBlock_down(in_channels=hidden_channels[1], out_channels=hidden_channels[2], embedding_dim=embedding_dim),
        down2 = Conv((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),

        conv_next_down3 = ConvNextBlock_down(in_channels=hidden_channels[2], out_channels=hidden_channels[3], embedding_dim=embedding_dim),
        down3 = Conv((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),

        # # Down-sampling path - using conditioning for down-scaling
        # conv_next_down1 = ConvNextBlock_up(in_channels=in_channels, out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=in_channels),
        # down1 = Conv((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),
        cond1_down = Chain(
            Conv((3, 3), cond_channels => hidden_channels[1], leakyrelu, pad=(1,1)),
            Conv((4, 4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2, 2)),
        ),

        # conv_next_down2 = ConvNextBlock_up(in_channels=hidden_channels[1], out_channels=hidden_channels[2], embedding_dim=embedding_dim, cond_channels=hidden_channels[1]),
        # down2 = Conv((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),
        cond2_down = Chain(
            Conv((3, 3), hidden_channels[1] => hidden_channels[2], leakyrelu, pad=(1,1)),
            Conv((4, 4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2, 2)),
        ),

        # conv_next_down3 = ConvNextBlock_up(in_channels=hidden_channels[2], out_channels=hidden_channels[3], embedding_dim=embedding_dim, cond_channels=hidden_channels[2]),
        # down3 = Conv((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),
        cond3_down = Chain(
            Conv((3, 3), hidden_channels[2] => hidden_channels[3], leakyrelu, pad=(1,1)),
            Conv((4, 4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2, 2)),
        ),

        # Bottom layer
        bottom = ConvNextBlock_down(in_channels=hidden_channels[3], out_channels=hidden_channels[3], embedding_dim=embedding_dim),

        # # Bottom layer with conditioning
        # bottom = ConvNextBlock_up(in_channels=hidden_channels[3], out_channels=hidden_channels[3], embedding_dim=embedding_dim, cond_channels=hidden_channels[3]),

        condup1 = Conv((3,3), hidden_channels[2] => 2 * hidden_channels[3], leakyrelu, pad=(1,1)),
        condup2 = Conv((3,3), hidden_channels[1] => 2 * hidden_channels[2], leakyrelu, pad=(1,1)),
        condup3 = Conv((3,3), cond_channels => 2 * hidden_channels[1], leakyrelu, pad=(1,1)),

        # Up-sampling path
        up3  = ConvTranspose((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),
        conv_next_up3 = ConvNextBlock_up(in_channels=2*hidden_channels[3], out_channels=hidden_channels[2], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[3]),

        up2 = ConvTranspose((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),
        conv_next_up2 = ConvNextBlock_up(in_channels=2*hidden_channels[2], out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[2]),

        up1 = ConvTranspose((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),
        conv_next_up1 = ConvNextBlock_up(in_channels=2*hidden_channels[1], out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[1]),

        # Output layer
        final_conv = Conv((1, 1), hidden_channels[1] => out_channels, use_bias=false),

    ) do x
        x, pars, cond = x 
        # size x: (32,32,embedding_dim,batch_size), size pars: (embedding_dim, batch_size), size cond: (32,32,embedding_dim,batch_size)

        # Down-sampling path - using no conditioning for the upscaling part. 
        skip_1 = conv_next_down1((x, pars)) # size: (32,32,hidden_channels[1],batch_size)
        x = down1(skip_1) # size: (16,16,hidden_channels[1],batch_size)

        skip_2 = conv_next_down2((x, pars)) # size: (16,16,hidden_channels[2],batch_size)
        x = down2(skip_2) # size: (8,8,hidden_channels[2],batch_size)

        skip_3 = conv_next_down3((x, pars)) # size: (8,8,hidden_channels[3], batch_size)
        x = down3(skip_3) # size: (4,4,hidden_channels[3], batch_size)

        # # Down-sampling path - using conditioning for the upscaling part
        # skip_1 = conv_next_down1((x, pars, cond)) # size: (32,32,hidden_channels[1],batch_size)
        # x = down1(skip_1) # size: (16,16,hidden_channels[1],batch_size)

        cond1d = cond1_down(cond) # size: (16,16,hidden_channels[1], batch_size)
        # skip_2 = conv_next_down2((x, pars, cond1d)) # size: (16,16,hidden_channels[2],batch_size)
        # x = down2(skip_2) # size: (8,8,hidden_channels[2],batch_size)

        cond2d = cond2_down(cond1d) # size: (8,8,hidden_channels[2], batch_size)
        # skip_3 = conv_next_down3((x, pars, cond)) # size: (8,8,hidden_channels[3], batch_size)
        # x = down3(skip_3) # size: (4,4,hidden_channels[3], batch_size)

        # Bottom layer
        x = bottom((x, pars)) # size: (4,4,hidden_channels[3],32)

        # # Bottom layer with conditioning
        # cond3d = cond3_down(cond2d) # size: (4,4,hidden_channels[3], batch_size)
        # x = bottom((x, pars, cond3d))

        # Generate different sizes for conditioning for the upscaling part
        cond_1 = condup1(cond2d) # size: (8,8,2*hidden_channels[3], batch_size)
        cond_2 = condup2(cond1d) # size; (16,16,2*hidden_channels[2],batch_size) 
        cond_3 = condup3(cond) # size; (32,32,2*hidden_channels[1],batch_size)

        x = up3(x) # size: (8,8,hidden_channels[3], batch_size)
        x = cat(x, skip_3, dims=3) # size: (8,8,2*hidden_channels[3],batch_size) 
        x = conv_next_up3((x, pars, cond_1)) # size: (8,8,hidden_channels[2], batch_size)

        x = up2(x) # size: (16,16,hidden_channels[2],batch_size)
        x = cat(x, skip_2, dims=3) # size: (16,16,2*hidden_channels[2],batch_size)
        x = conv_next_up2((x, pars, cond_2)) # size: (16,16,hidden_channels[1],batch_size)

        x = up1(x) # size: (32,32,hidden_channels[1],batch_size)
        x = cat(x, skip_1, dims=3) # size: (32,32,2*hidden_channels[1],batch_size)
        x = conv_next_up1((x, pars, cond_3)) # size: (32,32,hidden_channels[1],batch_size)

        @return final_conv(x) # (32,32,1,batch_size)
    end
end

# Full U-Net model with time embedding
function build_full_unet(embedding_dim = 8, hidden_channels = [16, 32, 64], t_pars_embedding_dim = 8)
    return @compact(
        conv_in = Conv((3, 3), 2 => embedding_dim, leakyrelu, pad=(1,1)),
        u_net = UNet(embedding_dim, 2, hidden_channels, embedding_dim, embedding_dim), 
        t_embedding = Chain(
            t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, t_pars_embedding_dim),
            Lux.Dense(t_pars_embedding_dim => embedding_dim),
            NNlib.gelu,
            Lux.Dense(embedding_dim => embedding_dim),
            NNlib.gelu,
          )
    ) do x
        I_sample, t_sample, cond = x # size I_sample: (32,32,1,batch_size)/(128,128,2,batch_size), t_sample: (1,1,1,batch_size), cond: (32,32,1,batch_size)/(128,128,2,batch_size)
        I_sample_phys = ifft(I_sample, (1,2))
        I_sample_real, I_sample_imag = real(I_sample_phys), imag(I_sample_phys)
        I_sample_real_stand = (I_sample_real .- mean(I_sample_real)) ./ std(I_sample_real)
        I_sample_imag_stand = (I_sample_imag .- mean(I_sample_imag)) ./ std(I_sample_imag)

        cond_phys = ifft(cond, (1,2))
        cond_real, cond_imag = real(cond_phys), imag(cond_phys)
        cond_real_stand = (cond_real .- mean(cond_real)) ./ std(cond_real)
        cond_imag_stand = (cond_imag .- mean(cond_imag)) ./ std(cond_imag)
        
        x_real = conv_in(I_sample_real_stand) # size: (32, 32, embedding_dim, batch_size)/(128,128,embedding_dim,batch_size)
        x_imag = conv_in(I_sample_imag_stand)
        cond_in_real = conv_in(cond_real) # size: (32, 32, embedding_dim, batch_size)/(128,128,embedding_dim,batch_size)
        cond_in_imag = conv_in(cond_imag)

        t_sample_embedded = t_embedding(t_sample) # size:(embedding_dim, batch_size)

        u_net_output_real = u_net((x_real, t_sample_embedded, cond_in_real))
        u_net_output_imag = u_net((x_imag, t_sample_embedded, cond_in_imag))

        # Combine real and imaginary outputs back into a complex result
        u_net_output_phys = u_net_output_real .+ im .* u_net_output_imag

        # Convert result back to spectral space
        u_net_output_spectral = fft(u_net_output_phys, (1,2))

        @return u_net_output_spectral
    end
end
