using Lux
using Random
using NNlib
using LuxCUDA
using FFTW
using CUDA

z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)
ArrayType = CUDA.functional() ? CuArray : Array
CUDA.allowscalar(false)


function gpu_attention_broadcast(query, key, value)
    hw, d, b = size(query)
    _, c, _ = size(value)
    output = zeros(Float32, hw, c, b)

    for i in 1:b
        q_i = query[:, :, i] 
        k_i = key[:, :, i] 
        v_i = value[:, :, i]
        attn_weights = softmax(q_i * k_i' ; dims=1)
        output[:, :, i] .= attn_weights * v_i
    end

    return output
end

function SelfAttentionBlock(;
    in_channels::Int64,
)
    @compact(
        query_proj = Conv((1, 1), in_channels => in_channels ÷ 8),
        key_proj = Conv((1, 1), in_channels => in_channels ÷ 8),
        value_proj = Conv((1, 1), in_channels => in_channels),
        proj_out = Conv((1, 1), in_channels => in_channels),
    ) do x
        h, w, c, b = size(x)
        query = Array(reshape(query_proj(x), h * w, c ÷ 8, b)) # (h, w, c/8, batch_size) -> (hw, c/8, batch_size)
        key = Array(reshape(key_proj(x), h * w, c ÷ 8, b))  # (h, w, c/8, batch_size) -> (hw, c/8, batch_size)
        value = Array(reshape(value_proj(x), h * w, c, b))  # (h, w, c, batch_size) -> (hw, c, batch_size)

        # Compute attention weights
        attention = gpu_attention_broadcast(query, key, value) # (hw, c, batch_size)
        # println(" compute attention ")
        # (c/8, hw, batch_size) x  (c/8, hw, batch_size) -> Attention: (hw, hw, batch_size)

        out = CuArray(reshape(attention, h, w, c, b)) # Reshape to (height, width, channels, batch_size)
        println("convert to right size")

        # Apply output projection and residual connection
        return proj_out(out) .+ x  # Residual connection
    end
end

# Sinusoidal embedding for the time
function sinusoidal_embedding(x, 
    min_freq::AbstractFloat, 
    max_freq::AbstractFloat, 
    embedding_dims::Int, 
    dev)
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> dev
    
    angular_speeds = reshape(2.0f0 * π * freqs, (1, 1, length(freqs), 1))
    
    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)

    return dropdims(embeddings, dims=(1, 2))
end

function ConvNextBlock_down_with_attention(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1,
)
    @compact(
        ds_conv = Conv((7, 7), in_channels => in_channels; pad=3),
        pars_mlp = Lux.Dense(embedding_dim => in_channels),
        attention = SelfAttentionBlock(in_channels=in_channels),  # Use the modular SelfAttentionBlock
        dropout1 = Dropout(0.3),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            Conv((3, 3), in_channels => in_channels * multiplier, pad=(1,1)),
            NNlib.gelu,
            Dropout(0.3),
            InstanceNorm(in_channels * multiplier),
            Lux.Conv((3, 3), in_channels * multiplier => out_channels, pad=(1,1))
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)
    ) do x
        x, pars = x
        h = ds_conv(x)
        h = dropout1(h)
        # println(" The size of the input of attention: ", size(h))

        # Apply attention
        h = attention(h)
        # println(" The size of the output of attention: ", size(h))

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

# ConvNextBlock Up with Self-Attention
function ConvNextBlock_up_with_attention(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1,
    cond_channels = 1,
)
    @compact(
        ds_conv = Conv((7, 7), in_channels => in_channels; pad=3),
        pars_mlp = Lux.Dense(embedding_dim => in_channels),
        attention = SelfAttentionBlock(in_channels=in_channels),
        cond_conv = Conv((1, 1), cond_channels => in_channels; pad=0),
        dropout1 = Dropout(0.1),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            Conv((3, 3), in_channels => in_channels * multiplier, pad=(1,1)),
            NNlib.gelu,
            Dropout(0.1),
            InstanceNorm(in_channels * multiplier),
            Lux.Conv((3, 3), in_channels * multiplier => out_channels, pad=(1,1))
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)
    ) do x
        x, pars, cond = x

        h = ds_conv(x)
        h = dropout1(h)

        # Add attention
        h = attention(h)

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
    hidden_channels = [16, 32, 64, 128],
    embedding_dim = 8,
    cond_channels = 1,
) # dropout_rate
    return @compact(
        # Down-sampling path - using no conditioning for down-scaling
        conv_next_down1 = ConvNextBlock_down_with_attention(in_channels=in_channels, out_channels=hidden_channels[1], embedding_dim=embedding_dim),
        down1 = Chain(
            Conv((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),
            Dropout(0.3)
        ),

        conv_next_down2 = ConvNextBlock_down_with_attention(in_channels=hidden_channels[1], out_channels=hidden_channels[2], embedding_dim=embedding_dim),
        down2 = Chain(
            Conv((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),
            Dropout(0.3)
        ),

        conv_next_down3 = ConvNextBlock_down_with_attention(in_channels=hidden_channels[2], out_channels=hidden_channels[3], embedding_dim=embedding_dim),
        down3 = Chain(
            Conv((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),
            Dropout(0.3)
        ),

        conv_next_down4 = ConvNextBlock_down_with_attention(in_channels=hidden_channels[3], out_channels=hidden_channels[4], embedding_dim=embedding_dim),
        down4 = Chain(
            Conv((4,4), hidden_channels[4] => hidden_channels[4]; pad=1, stride=(2,2)),
            Dropout(0.3)
        ),

        # # Down-sampling path - using conditioning for down-scaling
        # conv_next_down1 = ConvNextBlock_down(in_channels=in_channels, out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=in_channels),
        # down1 = Conv((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),
        cond1_down = Chain(
            Conv((3, 3), cond_channels => hidden_channels[1], leakyrelu, pad=(1,1)),
            Conv((4, 4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2, 2)),
        ),

        # conv_next_down2 = ConvNextBlock_down(in_channels=hidden_channels[1], out_channels=hidden_channels[2], embedding_dim=embedding_dim, cond_channels=hidden_channels[1]),
        # down2 = Conv((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),
        cond2_down = Chain(
            Conv((3, 3), hidden_channels[1] => hidden_channels[2], leakyrelu, pad=(1,1)),
            Conv((4, 4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2, 2)),
        ),

        # conv_next_down3 = ConvNextBlock_down(in_channels=hidden_channels[2], out_channels=hidden_channels[3], embedding_dim=embedding_dim, cond_channels=hidden_channels[2]),
        # down3 = Conv((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),
        cond3_down = Chain(
            Conv((3, 3), hidden_channels[2] => hidden_channels[3], leakyrelu, pad=(1,1)),
            Conv((4, 4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2, 2)),
        ),

        # Bottom layer
        bottom = ConvNextBlock_down_with_attention(in_channels=hidden_channels[4], out_channels=hidden_channels[4], embedding_dim=embedding_dim), #, cond_channels=hidden_channels[3]),

        # # Bottom layer with conditioning
        # bottom = ConvNextBlock_up(in_channels=hidden_channels[3], out_channels=hidden_channels[3], embedding_dim=embedding_dim, cond_channels=hidden_channels[3]),

        condup1 = Conv((3,3), hidden_channels[2] => 2 * hidden_channels[3], leakyrelu, pad=(1,1)),
        condup2 = Conv((3,3), hidden_channels[1] => 2 * hidden_channels[2], leakyrelu, pad=(1,1)),
        condup3 = Conv((3,3), cond_channels => 2 * hidden_channels[1], leakyrelu, pad=(1,1)),
        condup4 = Conv((3,3), hidden_channels[3] => 2*hidden_channels[4], leakyrelu, pad=(1,1)),

        up4  = Chain(
            ConvTranspose((4,4), hidden_channels[4] => hidden_channels[4]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up4 = ConvNextBlock_up_with_attention(in_channels=2*hidden_channels[4], out_channels=hidden_channels[3], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[4]),

        # Up-sampling path
        up3  = Chain(
            ConvTranspose((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up3 = ConvNextBlock_up_with_attention(in_channels=2*hidden_channels[3], out_channels=hidden_channels[2], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[3]),

        up2 = Chain(
            ConvTranspose((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up2 = ConvNextBlock_up_with_attention(in_channels=2*hidden_channels[2], out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[2]),

        up1 = Chain(
            ConvTranspose((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up1 = ConvNextBlock_up_with_attention(in_channels=2*hidden_channels[1], out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[1]),

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

        skip_4  = conv_next_down4((x, pars)) # size: (4,4,hidden_channels[4], batch_size)
        x = down4(skip_4) # size: (2,2,hidden_channels[4],batch_size)

        # Down-sampling path - using conditioning for the upscaling part
        # skip_1 = conv_next_down1((x, pars, cond)) # size: (32,32,hidden_channels[1],batch_size)
        # x = down1(skip_1) # size: (16,16,hidden_channels[1],batch_size)

        cond1d = cond1_down(cond) # size: (16,16,hidden_channels[1], batch_size)

        # skip_2 = conv_next_down2((x, pars, cond1d)) # size: (16,16,hidden_channels[2],batch_size)
        # x = down2(skip_2) # size: (8,8,hidden_channels[2],batch_size)

        cond2d = cond2_down(cond1d) # size: (8,8,hidden_channels[2], batch_size)

        # skip_3 = conv_next_down3((x, pars, cond2d)) # size: (8,8,hidden_channels[3], batch_size)
        # x = down3(skip_3) # size: (4,4,hidden_channels[3], batch_size)

        cond3d = cond3_down(cond2d) # size: (4,4,hidden_channels[3], batch_size)

        # Bottom layer
        x = bottom((x, pars)) # size: (2,2,hidden_channels[4],batch_size)

        # # Bottom layer with conditioning
        # cond3d = cond3_down(cond2d) # size: (4,4,hidden_channels[3], batch_size)
        # x = bottom((x, pars, cond3d))

        # Generate different sizes for conditioning for the upscaling part
        cond_1 = condup4(cond3d) # size: (4,4,2*hidden_channels[4],batch_size)
        cond_2 = condup1(cond2d) # size: (8,8,2*hidden_channels[3], batch_size)
        cond_3 = condup2(cond1d) # size; (16,16,2*hidden_channels[2],batch_size) 
        cond_4 = condup3(cond) # size; (32,32,2*hidden_channels[1],batch_size)

        x = up4(x) # size: (4,4,hidden_channels[4], batch_size)
        x = cat(x, skip_4, dims=3) # size: (4,4,2*hidden_channels[4], batch_size)
        x = conv_next_up4((x, pars, cond_1)) # size: 

        x = up3(x) # size: (8,8,hidden_channels[3], batch_size)
        x = cat(x, skip_3, dims=3) # size: (8,8,2*hidden_channels[3],batch_size) 
        x = conv_next_up3((x, pars, cond_2)) # size: (8,8,hidden_channels[2], batch_size)

        x = up2(x) # size: (16,16,hidden_channels[2],batch_size)
        # println(" check 16")
        x = cat(x, skip_2, dims=3) # size: (16,16,2*hidden_channels[2],batch_size)
        # println("check 17")
        x = conv_next_up2((x, pars, cond_3)) # size: (16,16,hidden_channels[1],batch_size)
        # println("check 18")

        x = up1(x) # size: (32,32,hidden_channels[1],batch_size)
        # println("check 19")
        x = cat(x, skip_1, dims=3) # size: (32,32,2*hidden_channels[1],batch_size)
        # println("check 20")
        x = conv_next_up1((x, pars, cond_4)) # size: (32,32,hidden_channels[1],batch_size)
        # println("check 21")

        @return final_conv(x) # (32,32,1,batch_size)
    end
end

# Full U-Net model with time embedding
function build_full_unet(embedding_dim = 8, hidden_channels = [16, 32, 64, 128], t_pars_embedding_dim = 8; dev) #, dropout_rate=0.1)
    return @compact(
        conv_in = Conv((3, 3), 2 => embedding_dim, leakyrelu, pad=(1,1)),
        u_net = UNet(embedding_dim, 2, hidden_channels, embedding_dim, embedding_dim), #, dropout_rate), 
        t_embedding = Chain(
            t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, t_pars_embedding_dim, dev),
            Lux.Dense(t_pars_embedding_dim => embedding_dim),
            NNlib.gelu,
            Lux.Dense(embedding_dim => embedding_dim),
            NNlib.gelu,
          )
    ) do x
        I_sample, t_sample, cond = x # size I_sample: (32,32,1,batch_size)/(128,128,2,batch_size), t_sample: (1,1,1,batch_size), cond: (32,32,1,batch_size)/(128,128,2,batch_size)
        
        x = conv_in(I_sample) # size: (32, 32, embedding_dim, batch_size)/(128,128,embedding_dim,batch_size)
        cond_in = conv_in(cond) # size: (32, 32, embedding_dim, batch_size)/(128,128,embedding_dim,batch_size)
        t_sample_embedded = t_embedding(t_sample) # size:(embedding_dim, batch_size)

        u_net_output = u_net((x, t_sample_embedded, cond_in))

        @return u_net_output
    end
end