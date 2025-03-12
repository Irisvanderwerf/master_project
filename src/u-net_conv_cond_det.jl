using Lux
using Random
using NNlib
using LuxCUDA
using FFTW
using CUDA

z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)
ArrayType = CUDA.functional() ? CuArray : Array
CUDA.allowscalar(false)

function ConvNextBlock_down_det(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1 
)
    @compact(
        ds_conv = ConvPeriodicLayer((7,7), in_channels, in_channels; pad=(3,3)),
        dropout1 = Dropout(0.1),
        conv_net = Chain(
            Lux.BatchNorm(in_channels),
            ConvPeriodicLayer((3, 3), in_channels, (in_channels * multiplier); pad=(1,1)),
            NNlib.gelu,
            Dropout(0.1),
            BatchNorm(in_channels * multiplier),
            ConvPeriodicLayer((3, 3), (in_channels * multiplier), out_channels; pad=(1,1))
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)
    ) do x
        h = ds_conv(x)
        h = dropout1(h)
        h = conv_net(h)
        @return h .+ res_conv(x)
    end
end

function ConvNextBlock_up_det(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1
)
    @compact(
        ds_conv = ConvPeriodicLayer((7,7), in_channels, in_channels; pad=(3,3)),
        dropout1 = Dropout(0.1), 
        conv_net = Chain(
            Lux.BatchNorm(in_channels),
            ConvPeriodicLayer((3, 3), in_channels, (in_channels * multiplier); pad=(1,1)),
            NNlib.gelu,
            Dropout(0.1),
            BatchNorm(in_channels * multiplier),
            ConvPeriodicLayer((3, 3), (in_channels * multiplier), out_channels; pad=(1,1)), 
        ),
        res_conv = Conv((1, 1), in_channels => out_channels; pad=0)
    ) do x
        h = ds_conv(x)
        h = dropout1(h)
        h = conv_net(h)
        @return h .+ res_conv(x)
    end
end

function BottomLayerWithAttention_det(;
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
            Lux.BatchNorm(in_channels),
            ConvPeriodicLayer((3, 3), in_channels, (in_channels * multiplier); pad=(1,1)),
            NNlib.gelu,
            Dropout(0.1),
            BatchNorm(in_channels * multiplier),
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

function UNet_det(
    in_channels = 1,
    out_channels = 1,
    hidden_channels = [16, 32, 64, 128],
    embedding_dim = 8
)
    return @compact(
        conv_next_down1 = ConvNextBlock_down_det(in_channels=in_channels, out_channels=hidden_channels[1]), 
        down1 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[1], hidden_channels[1]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        conv_next_down2 = ConvNextBlock_down_det(in_channels=hidden_channels[1], out_channels=hidden_channels[2]), 
        down2 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[2], hidden_channels[2]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        conv_next_down3 = ConvNextBlock_down_det(in_channels=hidden_channels[2], out_channels=hidden_channels[3]), 
        down3 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[3], hidden_channels[3]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        conv_next_down4 = ConvNextBlock_down_det(in_channels=hidden_channels[3], out_channels=hidden_channels[4]),
        down4 = Chain(
            ConvPeriodicLayer((4,4), hidden_channels[4], hidden_channels[4]; stride=(2,2), pad=(1,1)),
            Dropout(0.1)
        ),
        bottom = ConvNextBlock_down_det(in_channels=hidden_channels[4], out_channels=hidden_channels[4]),

        up4 = Chain(
            ConvTranspose((4,4), hidden_channels[4] => hidden_channels[4]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up4 = ConvNextBlock_up_det(in_channels=2*hidden_channels[4], out_channels=hidden_channels[3]), 
        up3  = Chain(
            ConvTranspose((4,4), hidden_channels[3] => hidden_channels[3]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up3 = ConvNextBlock_up_det(in_channels=2*hidden_channels[3], out_channels=hidden_channels[2]), 
        up2 = Chain(
            ConvTranspose((4,4), hidden_channels[2] => hidden_channels[2]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up2 = ConvNextBlock_up_det(in_channels=2*hidden_channels[2], out_channels=hidden_channels[1]),
        up1 = Chain(            
            ConvTranspose((4,4), hidden_channels[1] => hidden_channels[1]; pad=1, stride=(2,2)),
            Dropout(0.1)
        ),
        conv_next_up1 = ConvNextBlock_up_det(in_channels=2*hidden_channels[1], out_channels=hidden_channels[1]),
        final_conv = Conv((1, 1), hidden_channels[1] => out_channels, use_bias=false),
    ) do x
        skip_1 = conv_next_down1((x))
        x = down1(skip_1)
        skip_2 = conv_next_down2((x)) 
        x = down2(skip_2)
        skip_3 = conv_next_down3((x))
        x = down3(skip_3)
        skip_4  = conv_next_down4((x))
        x = down4(skip_4)
        x = bottom((x)) 
        x = up4(x)
        x = cat(x, skip_4, dims=3) 
        x = conv_next_up4((x))
        x = up3(x) 
        x = cat(x, skip_3, dims=3)
        x = conv_next_up3((x))
        x = up2(x)
        x = cat(x, skip_2, dims=3)
        x = conv_next_up2((x)) 
        x = up1(x) 
        x = cat(x, skip_1, dims=3) 
        x = conv_next_up1((x)) 
        @return final_conv(x)
    end
end

function build_full_unet_det(embedding_dim = 8, hidden_channels = [16, 32, 64, 128]) 
    return @compact(
        conv_in = ConvPeriodicLayer((3, 3), 2, embedding_dim; pad=(1,1), activation=NNlib.leakyrelu),
        u_net = UNet_det(embedding_dim, 2, hidden_channels, embedding_dim), 
    ) do x
        I_sample = x 
        x = conv_in(I_sample)
        u_net_output = u_net((x)) 

        @return u_net_output
    end
end
