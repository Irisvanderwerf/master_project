using Lux
using Random
using NNlib
using LuxCUDA
using FFTW
using CUDA

z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)
ArrayType = CUDA.functional() ? CuArray : Array
CUDA.allowscalar(false)

function periodic_padding_gpu(x, pad)
    pad_h, pad_w = pad
    H, W, C, B = size(x)

    top_pad = x[end-pad_h+1:end, :, :, :] 
    bottom_pad = x[1:pad_h, :, :, :]  
    padded_x = vcat(top_pad, x, bottom_pad)

    left_pad = padded_x[:, end-pad_w+1:end, :, :] 
    right_pad = padded_x[:, 1:pad_w, :, :]       
    padded_x = hcat(left_pad, padded_x, right_pad)

    return padded_x
end

function ConvPeriodicLayer(
    kernel_size::Tuple{Int, Int}, 
    in_channels::Int, 
    out_channels::Int; 
    stride::Tuple{Int, Int} = (1,1), 
    pad::Tuple{Int, Int} = (1,1), 
    activation::Function = identity  
)
    @compact(
        conv = Conv(kernel_size, in_channels => out_channels; stride=stride, pad=0)  
    ) do x
        padded_x = periodic_padding_gpu(x, pad) 
        conv_out = conv(padded_x)                
        @return activation(conv_out)              
    end
end

function SelfAttentionBlock(;
    in_channels::Int64
    )
    @compact(
        query_proj = Conv((1, 1), in_channels => in_channels ÷ 8),
        key_proj = Conv((1, 1), in_channels => in_channels ÷ 8),
        value_proj = Conv((1, 1), in_channels => in_channels),
        proj_out = Conv((1, 1), in_channels => in_channels),
    ) do x
        h, w, c, b = size(x)

        query = reshape(query_proj(x) |> gpu_device(), h * w, c ÷ 8, b)
        key = reshape(key_proj(x) |> gpu_device(), h * w, c ÷ 8, b)
        value = reshape(value_proj(x) |> gpu_device(), h * w, c, b)

        scale = sqrt(size(query, 2))
        attn_scores = batched_mul(query, permutedims(key, (2, 1, 3))) ./ scale
        attn_scores = attn_scores .- maximum(attn_scores, dims=2)  
        attn_weights = softmax(attn_scores; dims=2)

        attention = batched_mul(attn_weights, value)

        out = reshape(attention, h, w, c, b) |> gpu_device()
        @return proj_out(out) .+ x 
    end
end

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

function ConvNextBlock_down(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1,
)
    @compact(
        ds_conv = ConvPeriodicLayer((7,7), in_channels, in_channels; pad=(3,3)),
        pars_mlp = Lux.Dense(embedding_dim => in_channels),
        dropout1 = Dropout(0.1),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            ConvPeriodicLayer((3, 3), in_channels, (in_channels * multiplier); pad=(1,1)),
            NNlib.gelu,
            Dropout(0.1),
            InstanceNorm(in_channels * multiplier),
            ConvPeriodicLayer((3, 3), (in_channels * multiplier), out_channels; pad=(1,1))
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)
    ) do x
        x, pars = x
        h = ds_conv(x)
        h = dropout1(h)
        pars = pars_mlp(pars)
        pars = reshape(pars, 1, 1, size(pars)...)
        h = h .+ pars
        h = conv_net(h)
        @return h .+ res_conv(x)
    end
end

function ConvNextBlock_up(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1,
    cond_channels = 1,
)
    @compact(
        ds_conv = ConvPeriodicLayer((7,7), in_channels, in_channels; pad=(3,3)),
        pars_mlp = Lux.Dense(embedding_dim => in_channels),
        cond_conv = Conv((1, 1), cond_channels => in_channels; pad=0),
        dropout1 = Dropout(0.1), 
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            ConvPeriodicLayer((3, 3), in_channels, (in_channels * multiplier); pad=(1,1)),
            NNlib.gelu,
            Dropout(0.1),
            InstanceNorm(in_channels * multiplier),
            ConvPeriodicLayer((3, 3), (in_channels * multiplier), out_channels; pad=(1,1)), 
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)
    ) do x
        x, pars, cond_closure, cond_state = x
        h = ds_conv(x)
        h = dropout1(h)
        pars = pars_mlp(pars)
        pars = reshape(pars, 1, 1, size(pars)...)
        cond_closure = cond_conv(cond_closure)
        cond_state = cond_conv(cond_state)
        h = (h .+ pars) .* (cond_state .+ cond_closure)
        h = conv_net(h)
        @return h .+ res_conv(x)
    end
end

function BottomLayerWithAttention(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1,
    embedding_dim::Int = 1
)
    @compact(
        ds_conv = ConvPeriodicLayer((7, 7), in_channels, in_channels; pad=(3,3)),
        pars_mlp = Lux.Dense(embedding_dim => in_channels),
        dropout1 = Dropout(0.1),
        attention = SelfAttentionBlock(in_channels=in_channels), 
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            ConvPeriodicLayer((3, 3), in_channels, (in_channels * multiplier); pad=(1,1)),
            NNlib.gelu,
            Dropout(0.1),
            InstanceNorm(in_channels * multiplier),
            ConvPeriodicLayer((3, 3), (in_channels * multiplier), out_channels; pad=(1,1))
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)
    ) do x
        x, pars = x
        h = ds_conv(x)
        h = dropout1(h)
        h = attention(h)
        pars = pars_mlp(pars)
        pars = reshape(pars, 1, 1, size(pars)...)
        h = h .+ pars
        h = conv_net(h)
        @return h .+ res_conv(x)
    end
end

function UNet(
    in_channels = 1,
    out_channels = 1,
    hidden_channels = [16, 32, 64, 128],
    embedding_dim = 8,
    cond_channels = 1
)
    return @compact(
        conv_next_down1 = ConvNextBlock_down(in_channels=in_channels, out_channels=hidden_channels[1], embedding_dim=embedding_dim),
        down1 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[1], hidden_channels[1]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        conv_next_down2 = ConvNextBlock_down(in_channels=hidden_channels[1], out_channels=hidden_channels[2], embedding_dim=embedding_dim),
        down2 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[2], hidden_channels[2]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        conv_next_down3 = ConvNextBlock_down(in_channels=hidden_channels[2], out_channels=hidden_channels[3], embedding_dim=embedding_dim),
        down3 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[3], hidden_channels[3]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        conv_next_down4 = ConvNextBlock_down(in_channels=hidden_channels[3], out_channels=hidden_channels[4], embedding_dim=embedding_dim),
        down4 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[4], hidden_channels[4]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        # bottom = BottomLayerWithAttention(in_channels=hidden_channels[4], out_channels=hidden_channels[4], embedding_dim=embedding_dim),
        bottom = ConvNextBlock_down(in_channels=hidden_channels[4], out_channels=hidden_channels[4], embedding_dim=embedding_dim), 
        condup1 = Chain(
            ConvPeriodicLayer((3, 3), (2 * hidden_channels[3]), (2 * hidden_channels[4]); pad=(1,1), activation=NNlib.leakyrelu), 
            ConvPeriodicLayer((4, 4), (2 * hidden_channels[4]), (2 * hidden_channels[4]); stride=(2,2), pad=(1,1)),
        ),
        condup2 = Chain(
            ConvPeriodicLayer((3, 3), (2 * hidden_channels[2]), (2 * hidden_channels[3]); pad=(1,1), activation=NNlib.leakyrelu), 
            ConvPeriodicLayer((4, 4), (2 * hidden_channels[3]), (2 * hidden_channels[3]); stride=(2,2), pad=(1,1)),
        ),
        condup3 = Chain(
            ConvPeriodicLayer((3, 3), (2 * hidden_channels[1]), (2 * hidden_channels[2]); pad=(1,1), activation=NNlib.leakyrelu), 
            ConvPeriodicLayer((4, 4), (2 * hidden_channels[2]), (2 * hidden_channels[2]); stride=(2,2), pad=(1,1)),
        ),
        condup4 = ConvPeriodicLayer((3,3), cond_channels, (2 * hidden_channels[1]); pad=(1,1), activation=NNlib.leakyrelu),

        up4 = Chain(
            ConvTranspose((4,4), hidden_channels[4] => hidden_channels[4]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up4 = ConvNextBlock_up(in_channels=2*hidden_channels[4], out_channels=hidden_channels[3], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[4]),
        up3  = Chain(
            ConvTranspose((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up3 = ConvNextBlock_up(in_channels=2*hidden_channels[3], out_channels=hidden_channels[2], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[3]),
        up2 = Chain(
            ConvTranspose((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up2 = ConvNextBlock_up(in_channels=2*hidden_channels[2], out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[2]),
        up1 = Chain(            
            ConvTranspose((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up1 = ConvNextBlock_up(in_channels=2*hidden_channels[1], out_channels=hidden_channels[1], embedding_dim=embedding_dim, cond_channels=2*hidden_channels[1]),
        final_conv = Conv((1, 1), hidden_channels[1] => out_channels, use_bias=false),
    ) do x
        x, pars, cond_closure, cond_state = x
        skip_1 = conv_next_down1((x, pars))
        x = down1(skip_1)
        skip_2 = conv_next_down2((x, pars))
        x = down2(skip_2)
        skip_3 = conv_next_down3((x, pars))
        x = down3(skip_3)
        skip_4  = conv_next_down4((x, pars))
        x = down4(skip_4)
        x = bottom((x, pars))
        cond_closure_4 = condup4(cond_closure)
        cond_closure_3 = condup3(cond_closure_4)
        cond_closure_2 = condup2(cond_closure_3) 
        cond_closure_1 = condup1(cond_closure_2) 
        cond_state_4 = condup4(cond_state)
        cond_state_3 = condup3(cond_state_4)
        cond_state_2 = condup2(cond_state_3) 
        cond_state_1 = condup1(cond_state_2) 
        x = up4(x)
        x = cat(x, skip_4, dims=3) 
        x = conv_next_up4((x, pars, cond_closure_1, cond_state_1))  
        x = up3(x) 
        x = cat(x, skip_3, dims=3)
        x = conv_next_up3((x, pars, cond_closure_2, cond_state_2)) 
        x = up2(x)
        x = cat(x, skip_2, dims=3)
        x = conv_next_up2((x, pars, cond_closure_3, cond_state_3)) 
        x = up1(x) 
        x = cat(x, skip_1, dims=3) 
        x = conv_next_up1((x, pars, cond_closure_4, cond_state_4))
        @return final_conv(x)
    end
end

function build_full_unet(embedding_dim = 8, hidden_channels = [16, 32, 64, 128], t_pars_embedding_dim = 8; dev) 
    return @compact(
        conv_in = ConvPeriodicLayer((3, 3), 2, embedding_dim; pad=(1,1), activation=NNlib.leakyrelu),
        u_net = UNet(embedding_dim, 2, hidden_channels, embedding_dim, embedding_dim), 
        t_embedding = Chain(
            t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, t_pars_embedding_dim, dev),
            Lux.Dense(t_pars_embedding_dim => embedding_dim),
            NNlib.gelu,
            Lux.Dense(embedding_dim => embedding_dim),
            NNlib.gelu,
          )
    ) do x
        I_sample, t_sample, cond_closure, cond_state = x
        x = conv_in(I_sample)
        cond_closure_in = conv_in(cond_closure)
        cond_state_in = conv_in(cond_state)
        t_sample_embedded = t_embedding(t_sample)
        u_net_output = u_net((x, t_sample_embedded, cond_closure_in, cond_state_in))
        @return u_net_output
    end
end