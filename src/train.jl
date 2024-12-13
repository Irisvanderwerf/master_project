using Zygote
using Statistics
using Optimisers
using BSON
using CUDA
using LuxCUDA
using Plots
using LinearAlgebra

function initialize_or_load_model(model_name::String, network::Any, load_path::Union{String, Nothing} = nothing; dev, method::Symbol = :ODE, is_gaussian)
    if method == :ODE
        if isnothing(load_path)
            println("Initializing new models with name: $model_name")
            ps_drift, st_drift = Lux.setup(Random.default_rng(), network) |> dev
            opt_drift = Optimisers.setup(Adam(1.0e-4, (0.9f0, 0.99f0), 1e-10), ps_drift)
        else
            println("Loading models from path: $load_path")
            ps_drift, st_drift, opt_drift = load_model(load_path, is_gaussian; dev, method)
        end
        ps_denoiser, st_denoiser, opt_denoiser = nothing, nothing, nothing
        return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser
    elseif method == :SDE
        if isnothing(load_path)
            println("Initializing new models with name: $model_name")
            ps_drift, st_drift = Lux.setup(Random.default_rng(), network) |> dev
            opt_drift = Optimisers.setup(AdamW(1.0e-4, (0.9f0, 0.99f0), 1e-6), ps_drift)
            if !is_gaussian
                ps_denoiser, st_denoiser = Lux.setup(Random.default_rng(), network) |> dev
                opt_denoiser = Optimisers.setup(AdamW(1.0e-4, (0.9f0, 0.99f0), 1e-2), ps_drift)
            else
                ps_denoiser, st_denoiser, opt_denoiser = nothing, nothing, nothing
            end
        else
            if !is_gaussian
                println("Loading models from path: $load_path")
                ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser = load_model(load_path, is_gaussian; dev, method)
            else
                println("Loading models from path: $load_path")
                ps_drift, st_drift, opt_drift = load_model(load_path, is_gaussian; dev, method)
                ps_denoiser, st_denoiser, opt_denoiser = nothing, nothing, nothing
            end
        end
        return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser 
    else
        error("Invalid method. Choose either :ODE or :SDE.")
    end
end

# Function to load model parameters and optimizer states with structure checks
function load_model(file_path, is_gaussian; dev, method=:ODE)
    if method == :ODE
        if dev==cpu_device()
            data = BSON.load(file_path)
            ps_drift = data[:ps_drift]
            st_drift = data[:st_drift]
            opt_drift = data[:opt_drift]
            println("Loaded model and optimizer states (drift) from $file_path")
            return ps_drift, st_drift, opt_drift
        else
            data = BSON.load(file_path)
            ps_drift = data[:ps_drift_cpu]
            ps_drift = deepcopy(ps_drift) |> dev
            st_drift = data[:st_drift_cpu]
            st_drift = deepcopy(st_drift) |> dev
            opt_drift = data[:opt_drift_cpu]
            opt_drift = deepcopy(opt_drift) |> dev
            println("Loaded model and optimizer states (drift) from $file_path")
            return ps_drift, st_drift, opt_drift
        end
    elseif method == :SDE
        if dev==cpu_device()
            data = BSON.load(file_path)
            ps_drift = data[:ps_drift]
            st_drift = data[:st_drift]
            opt_drift = data[:opt_drift]
            if !is_gaussian
                ps_denoiser = data[:ps_denoiser]
                st_denoiser = data[:st_denoiser]
                opt_denoiser = data[:opt_denoiser]
                println("Loaded model and optimizer states (drift and denoiser) from $file_path")
                return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser
            else
                println("Loaded model and optimizer states (drift) from $file_path")
                return ps_drift, st_drift, opt_drift
            end
        else
            data = BSON.load(file_path)
            ps_drift = data[:ps_drift_cpu]
            ps_drift = deepcopy(ps_drift) |> dev
            st_drift = data[:st_drift_cpu]
            st_drift = deepcopy(st_drift) |> dev
            opt_drift = data[:opt_drift_cpu]
            opt_drift = deepcopy(opt_drift) |> dev
            if !is_gaussian
                ps_denoiser = data[:ps_denoiser_cpu]
                ps_denoiser = deepcopy(ps_denoiser) |> dev
                st_denoiser = data[:st_denoiser_cpu]
                st_denoiser = deepcopy(st_denoiser) |> dev
                opt_denoiser = data[:opt_denoiser_cpu]
                opt_denoiser = deepcopy(opt_denoiser) |> dev
                println("Loaded model and optimizer states (drift and denoiser) from $file_path")
                return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser
            else
                println("Loaded model and optimizer states (drift) from $file_path")
                return ps_drift, st_drift, opt_drift
            end
        end
    end
end

function save_model(file_path, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser; dev, method=:ODE)
    if method == :ODE
        if dev==cpu_device()
            BSON.@save file_path ps_drift st_drift opt_drift
            println("Model and optimizer states (drift) saved to $file_path")
        else
            ps_drift_cpu = ps_drift |> cpu_device()
            st_drift_cpu = st_drift |> cpu_device()
            opt_drift_cpu = opt_drift |> cpu_device()
            BSON.@save file_path ps_drift_cpu st_drift_cpu opt_drift_cpu
            println("Model and optimizer states (drift) saved to $file_path on CPU.")
        end
    else
        if dev==cpu_device()
            BSON.@save file_path ps_drift st_drift opt_drift ps_denoiser st_denoiser opt_denoiser
            println("Model and optimizer states (drift) saved to $file_path")
        else
            ps_drift_cpu = ps_drift |> cpu_device()
            st_drift_cpu = st_drift |> cpu_device()
            opt_drift_cpu = opt_drift |> cpu_device()
            ps_denoiser_cpu = ps_denoiser |> cpu_device()
            st_denoiser_cpu = st_denoiser |> cpu_device()
            opt_denoiser_cpu = opt_denoiser |> cpu_device()
            BSON.@save file_path ps_drift_cpu st_drift_cpu opt_drift_cpu ps_denoiser_cpu st_denoiser_cpu opt_denoiser_cpu
            println("Model and optimizer states (drift) saved to $file_path on CPU.")
        end
    end
end

function get_minibatch_NS(images, batch_size, batch_index)
    start_index = (batch_index - 1) * batch_size + 1
    end_index = min(batch_index * batch_size, size(images, 4)) 
    minibatch = images[:,:,:,start_index:end_index]
    return minibatch 
end

function loss_fn(velocity, dI_dt_sample)
    loss = mean((velocity .- dI_dt_sample).^2)
    return loss
end

function train!(initial_train_images, train_images, train_labels, batch_size, num_epochs, ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser, velocity_cnn, target_test, initial_test, target_label_test, save_path, model_name; is_gaussian=false, method=:SDE, dev=gpu.device())
    initial_train_images = reshape(initial_train_images, size(initial_train_images, 1), size(initial_train_images, 2), size(initial_train_images, 3), size(initial_train_images, 4) * size(initial_train_images, 5))  
    train_images = reshape(train_images, size(train_images, 1), size(train_images, 2), size(train_images, 3), size(train_images, 4) * size(train_images, 5)) 
    train_labels = reshape(train_labels, size(train_labels, 1), size(train_labels, 2), size(train_labels, 3), size(train_labels, 4) * size(train_labels, 5))

    target_test_use = reshape(target_test, size(target_test, 1), size(target_test, 2), size(target_test, 3), size(target_test, 4) * size(target_test, 5))
    target_label_test_use = reshape(target_label_test, size(target_label_test, 1), size(target_label_test, 2), size(target_label_test, 3), size(target_label_test, 4) * size(target_label_test, 5))
    initial_test_use = reshape(initial_test, size(initial_test, 1), size(initial_test, 2), size(initial_test, 3), size(initial_test, 4) * size(initial_test, 5))
    
    num_samples = size(train_images, 4);
    num_batches =  ceil(Int, num_samples / batch_size);

    init_learning_rate = 1.0e-5
    min_learning_rate = 1.0e-7
  
    drift_losses = Float32[]
    denoiser_losses = Float32[]
    test_drift_losses = Float32[]
    test_denoiser_losses = Float32[]

    best_test_loss_drift = Inf;
    best_test_loss_denoiser = Inf;
    patience = 20;
    counter = 0;
    stop_training = false;

    for epoch in 1:num_epochs
        if !stop_training 
            println("Epoch $epoch")
            shuffled_indices = randperm(size(train_images, 4))
            train_images = train_images[:, :, :, shuffled_indices]
            train_labels = train_labels[:, :, :, shuffled_indices]
        
            new_learning_rate = min_learning_rate .+ 0.5f0 .* (init_learning_rate - min_learning_rate) .* (1 .+ cos.(epoch ./ num_epochs .* π))
            Optimisers.adjust!(opt_drift, new_learning_rate)
            if !is_gaussian && method == :SDE
                Optimisers.adjust!(opt_denoiser, new_learning_rate)
            end

            epoch_drift_loss = 0.0
            if !is_gaussian && method == :SDE
                epoch_denoiser_loss = 0.0
            end

            for batch_index in 1:num_batches-1
                initial_sample = Float32.(get_minibatch_NS(initial_train_images, batch_size, batch_index)) |> dev
                target_sample = Float32.(get_minibatch_NS(train_images, batch_size, batch_index)) |> dev  
                target_labels_sample = Float32.(get_minibatch_NS(train_labels, batch_size, batch_index)) |> dev 

                t_sample = Float32.(reshape(rand(Float32, batch_size), 1, 1, 1, batch_size)) |> dev
                z_sample = Float32.(randn(size(target_sample))) |> dev
                I_sample = Float32.(stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample)) |> dev

                loss_drift_closure = (ps_) -> begin
                    dI_dt_sample = Float32.(time_derivative_stochastic_interpolant(initial_sample, target_sample, z_sample, t_sample))
                    velocity, st_drift = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_sample), ps_, st_drift)
                    return loss_fn(velocity, dI_dt_sample), st_drift
                end

                (loss_drift, st_drift), pb_drift_f = Zygote.pullback(
                    p -> loss_drift_closure(p), ps_drift
                ); 

                epoch_drift_loss += loss_drift

                gs_drift = pb_drift_f((one(loss_drift), nothing))[1];
                opt_drift, ps_drift = Optimisers.update!(opt_drift, ps_drift, gs_drift)

                if !is_gaussian && method == :SDE
                    loss_denoiser_closure = (ps_) -> begin
                        denoiser, st_denoiser = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_sample), ps_, st_denoiser) 
                        return loss_fn(denoiser, z_sample), st_denoiser
                    end

                    (loss_denoiser, st_denoiser), pb_denoiser_f = Zygote.pullback(
                        p -> loss_denoiser_closure(p), ps_denoiser
                    ); 

                    epoch_denoiser_loss += loss_denoiser

                    gs_denoiser = pb_denoiser_f((one(loss_denoiser), nothing))[1];
                    opt_denoiser, ps_denoiser = Optimisers.update!(opt_denoiser, ps_denoiser, gs_denoiser)  
                end
            end
            epoch_drift_loss /= num_batches
            println("Epoch loss of the drift term: $epoch_drift_loss")
            push!(drift_losses, epoch_drift_loss)

            if !is_gaussian && method == :SDE
                epoch_denoiser_loss /= num_batches
                println("Epoch loss of the denoiser term: $epoch_denoiser_loss")
                push!(denoiser_losses, epoch_denoiser_loss)
            end

            test_drift_loss = 0.0
            if !is_gaussian && method == :SDE
                test_denoiser_loss = 0.0
            end

            num_test_samples = size(target_test_use, 4)

            test_target = Float32.(target_test_use) |> dev
            test_label = Float32.(target_label_test_use) |> dev
            test_t = Float32.(reshape(rand(Float32, num_test_samples), 1, 1, 1, num_test_samples)) |> dev
            test_z = Float32.(randn(size(test_target))) |> dev
            initial_test_sample = Float32.(randn(size(initial_test_use))) |> dev

            _st_drift = Lux.testmode(st_drift) |> dev
            if !is_gaussian && method == :SDE
                _st_denoiser = Lux.testmode(st_denoiser) |> dev
            end
        
            I_sample_test = Float32.(stochastic_interpolant(initial_test_sample, test_target, test_z, test_t)) 
            dI_dt_sample_test = Float32.(time_derivative_stochastic_interpolant(initial_test_sample, test_target, test_z, test_t)) 
            velocity_test, _ = Lux.apply(velocity_cnn, (I_sample_test, test_t, test_label), ps_drift, _st_drift) 
            test_drift_loss += loss_fn(velocity_test, dI_dt_sample_test)
            push!(test_drift_losses, test_drift_loss)
            println("Test loss for drift term: $test_drift_loss")

            if !is_gaussian && method == :SDE
                velocity, _ = Lux.apply(velocity_cnn, (I_sample_test, test_t, test_label), ps_denoiser, _st_denoiser)
                test_denoiser_loss += loss_fn(velocity, test_z)
                push!(test_denoiser_losses, test_denoiser_loss)
                println("Test loss for denoiser term: $test_denoiser_loss")
            end

            if !is_gaussian && method == :SDE
                if test_drift_loss < best_test_loss_drift && test_denoiser_loss < best_test_loss_denoiser
                    best_test_loss_drift = test_drift_loss 
                    best_test_loss_denoiser = test_denoiser_loss
                    counter = 0 
                elseif test_drift_loss < best_test_loss_drift
                    best_test_loss_drift = test_drift_loss
                    counter = 0
                elseif test_denoiser_loss < best_test_loss_denoiser
                    best_test_loss_denoiser = test_denoiser_loss
                    counter = 0
                else
                    counter += 1
                    if counter >= patience
                        println("Early stopping triggered")
                        stop_training = true;
                        num_finished_epochs = epoch;
                    end
                end
            else
                if test_drift_loss < best_test_loss_drift
                    best_test_loss_drift = test_drift_loss 
                    counter = 0  
                else
                    counter += 1
                    if counter >= patience
                        println("Early stopping triggered")
                        stop_training = true;
                        num_finished_epochs = epoch;
                    end
                end
            end
        end
    end

    if !stop_training
        num_finished_epochs = num_epochs;
    end
    p = plot(1:num_finished_epochs, drift_losses, label="Drift Training Loss", xlabel="Epoch", ylabel="Loss", title="Training and Test Loss", yscale=:log10)
    plot!(p, 1:num_finished_epochs, test_drift_losses, label="Drift Test Loss")
    if !is_gaussian && method == :SDE
        plot!(p, 1:num_finished_epochs, denoiser_losses, label="Denoiser Training Loss")
        plot!(p, 1:num_finished_epochs, test_denoiser_losses, label = "Denoiser Test Loss")
    end
    savefig(p, "figures/final_loss_plot_$(model_name).png")
    println("Final loss plot saved at $(save_path)/final_loss_plot.png")

    println("Training completed. Saving the final model")
    save_model("$save_path/$model_name.bson", ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser; dev, method=:SDE)
    return ps_drift, st_drift, opt_drift, ps_denoiser, st_denoiser, opt_denoiser
end

function train_deterministic!(velocity_cnn, ps_drift, st_drift, opt_drift, num_epochs, batch_size, train_images, train_labels, model_name, save_path, v_test, c_test; dev)
    train_images = reshape(train_images, size(train_images, 1), size(train_images, 2), size(train_images, 3), size(train_images, 4) * size(train_images, 5)) 
    train_labels = reshape(train_labels, size(train_labels, 1), size(train_labels, 2), size(train_labels, 3), size(train_labels, 4) * size(train_labels, 5))
    v_test_use = reshape(v_test, size(v_test, 1), size(v_test, 2), size(v_test, 3), size(v_test, 4) * size(v_test, 5))
    c_test_use = reshape(c_test, size(c_test, 1), size(c_test, 2), size(c_test, 3), size(c_test, 4) * size(c_test, 5))
    
    num_samples = size(train_images, 4);
    num_batches =  ceil(Int, num_samples / batch_size);
  
    init_learning_rate = 1.0e-3
    min_learning_rate = 1.0e-6
  
    drift_losses = Float32[]
    test_drift_losses = Float32[]

    best_test_loss_drift = Inf;
    patience = 10;
    counter = 0;
    stop_training = false;

    for epoch in 1:num_epochs
        if !stop_training 
            println("Epoch $epoch")
            shuffled_indices = randperm(size(train_images, 4))
            train_images = train_images[:, :, :, shuffled_indices]
            train_labels = train_labels[:, :, :, shuffled_indices]

            new_learning_rate = min_learning_rate .+ 0.5f0 .* (init_learning_rate - min_learning_rate) .* (1 .+ cos.(epoch ./ num_epochs .* π))
            Optimisers.adjust!(opt_drift, new_learning_rate)

            epoch_drift_loss = 0.0
            for batch_index in 1:num_batches-1
                target_sample = Float32.(get_minibatch_NS(train_images, batch_size, batch_index)) |> dev  
                target_labels_sample = Float32.(get_minibatch_NS(train_labels, batch_size, batch_index)) |> dev 
                t_sample = Float32.(fill(0, 1, 1, 1, batch_size)) |> dev
                initial_sample = target_labels_sample |> dev

                loss_drift_closure = (ps_) -> begin
                    closure, st_drift = Lux.apply(velocity_cnn, (initial_sample, t_sample), ps_, st_drift) #, target_labels_sample), ps_, st_drift)
                    return loss_fn(closure, target_sample), st_drift
                end

                (loss_drift, st_drift), pb_drift_f = Zygote.pullback(
                    p -> loss_drift_closure(p), ps_drift
                ); 

                epoch_drift_loss += loss_drift

                gs_drift = pb_drift_f((one(loss_drift), nothing))[1];
                opt_drift, ps_drift = Optimisers.update!(opt_drift, ps_drift, gs_drift)
            end
            epoch_drift_loss /= num_batches
            println("Epoch loss of the drift term: $epoch_drift_loss")
            push!(drift_losses, epoch_drift_loss)

            test_drift_loss = 0.0
            # trajectory = rand(1:4);
            # v_test_use = v_test[:,:,:,:,trajectory];
            # c_test_use = c_test[:,:,:,:,trajectory];
            num_samples = size(v_test_use, 4);

            test_target = Float32.(c_test_use) |> dev
            test_label = Float32.(v_test_use) |> dev
            test_initial_sample = test_label |> dev 
            test_t = Float32.(fill(0, 1, 1, 1, num_samples)) |> dev 
            _st_drift = Lux.testmode(st_drift) |> dev
        
            closure_test, _ = Lux.apply(velocity_cnn, (test_initial_sample, test_t), ps_drift, _st_drift) #, test_label), ps_drift, _st_drift) 
            test_drift_loss += loss_fn(closure_test, test_target)
            push!(test_drift_losses, test_drift_loss)
            println("Test loss for drift term: $test_drift_loss")


            if test_drift_loss < best_test_loss_drift
                best_test_loss_drift = test_drift_loss 
                counter = 0
            else
                counter += 1
                if counter >= patience
                    println("Early stopping triggered")
                    stop_training = true;
                    num_finished_epochs = epoch;
                end
            end
        end
    end

    if !stop_training
        num_finished_epochs = num_epochs;
    end

    p = plot(1:num_finished_epochs, drift_losses, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Training and Test Loss", yscale=:log10)
    plot!(p, 1:num_finished_epochs, test_drift_losses, label="Test Loss")
    savefig(p, "figures/final_loss_plot_$(model_name).png")
    println("Final loss plot saved at $(save_path)/final_loss_plot.png")

    println("Training completed. Saving the final model")
    save_model("$save_path/$model_name.bson", ps_drift, st_drift, opt_drift, nothing, nothing, nothing; dev, method=:ODE)
    return ps_drift, st_drift, opt_drift
end