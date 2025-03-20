using Lux

function sinusoidal_embedding(x, min_freq::AbstractFloat, max_freq::AbstractFloat, embedding_dims::Int, dev=gpu_device())
    if length(size(x)) != 4
        x = reshape(x, (1, 1, 1, size(x)[end]))
    end
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> dev
    angular_speeds = reshape(2.0f0 * π * freqs, (1, 1, length(freqs), 1))
    embeddings = cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims=3)
    return embeddings
end

function build_NN()
    return @compact(
        conv_in = Conv((3, 3), 1 => 32, leakyrelu, pad=(1,1)),
        conv_layers = Chain(
            Conv((3, 3), 64 => 64, leakyrelu, pad=(1,1)),   
            Conv((3, 3), 64 => 64, leakyrelu, pad=(1,1)),  
            Conv((3, 3), 64 => 64, leakyrelu, pad=(1,1)),  
            Conv((3, 3), 64 => 1, pad=1, use_bias=false),   
        ),
        t_embedding = t -> sinusoidal_embedding(t, 1.0f0, 1000.0f0, 32)
    ) do x
        I_sample, t_sample = x
        x = conv_in(I_sample)
        t_sample_reshaped = repeat(t_sample, 28, 28, 1, 1)
        t_sample_reshaped = t_embedding(t_sample_reshaped)
        x = cat(x, t_sample_reshaped, dims=3)
        @return conv_layers(x)
    end
end