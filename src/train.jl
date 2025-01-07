using Zygote
using Statistics
using Optimisers
using BSON
using CUDA
using LuxCUDA
using Plots
using LinearAlgebra
using Random

function initialize_or_load_model(model_name::String, network::Any, load_path::Union{String, Nothing} = nothing; dev)
    if isnothing(load_path)
        println("Initializing new models with name: $model_name")
        ps_drift, st_drift = Lux.setup(Random.default_rng(), network) |> dev
        opt_drift = Optimisers.setup(Adam(1.0e-4, (0.9f0, 0.99f0), 1e-6), ps_drift)
    else
        println("Loading models from path: $load_path")
        ps_drift, st_drift, opt_drift = load_model(load_path; dev)
    end
    return ps_drift, st_drift, opt_drift
end

function load_model(file_path; dev)
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

function save_model(file_path, ps_drift, st_drift, opt_drift; dev)
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
end


function train!(train_data, val_data, batch_size, num_epochs, ps_drift, st_drift, opt_drift, velocity_cnn, save_path, model_name, num_samples, eval_frequency, val_subset_size; dev=gpu.device())
    initial_train_images, train_images, train_labels_closure, train_labels_state = train_data.initial, train_data.target, train_data.closure, train_data.state;
    initial_test, target_test, target_label_closure_test, target_label_state_test = val_data.initial, val_data.target, val_data.closure, val_data.state;

    max_time_steps = size(initial_train_images, 4)
    if isnothing(num_samples) || num_samples > max_time_steps
        num_samples = max_time_steps 
    end

    initial_train_images = initial_train_images[:, :, :, 1:num_samples, :]
    train_images = train_images[:, :, :, 1:num_samples, :]
    train_labels_closure = train_labels_closure[:, :, :, 1:num_samples, :]
    train_labels_state = train_labels_state[:, :, :, 1:num_samples, :]

    initial_test = initial_test[:, :, :, 1:num_samples, :]
    target_test = target_test[:, :, :, 1:num_samples, :]
    target_label_closure_test = target_label_closure_test[:, :, :, 1:num_samples, :]
    target_label_state_test = target_label_state_test[:, :, :, 1:num_samples, :]

    initial_train_images = reshape(initial_train_images, size(initial_train_images, 1), size(initial_train_images, 2), size(initial_train_images, 3), size(initial_train_images, 4) * size(initial_train_images, 5))  
    train_images = reshape(train_images, size(train_images, 1), size(train_images, 2), size(train_images, 3), size(train_images, 4) * size(train_images, 5)) 
    train_labels_closure = reshape(train_labels_closure, size(train_labels_closure, 1), size(train_labels_closure, 2), size(train_labels_closure, 3), size(train_labels_closure, 4) * size(train_labels_closure, 5))
    train_labels_state = reshape(train_labels_state, size(train_labels_state, 1), size(train_labels_state, 2), size(train_labels_state, 3), size(train_labels_state, 4) * size(train_labels_state, 5))

    target_test = reshape(target_test, size(target_test, 1), size(target_test, 2), size(target_test, 3), size(target_test, 4) * size(target_test, 5))
    target_label_closure_test = reshape(target_label_closure_test, size(target_label_closure_test, 1), size(target_label_closure_test, 2), size(target_label_closure_test, 3), size(target_label_closure_test, 4) * size(target_label_closure_test, 5))
    target_label_state_test = reshape(target_label_state_test, size(target_label_state_test, 1), size(target_label_state_test, 2), size(target_label_state_test, 3), size(target_label_state_test, 4) * size(target_label_state_test, 5))
    initial_test = reshape(initial_test, size(initial_test, 1), size(initial_test, 2), size(initial_test, 3), size(initial_test, 4) * size(initial_test, 5))

    total_test_samples = size(initial_test, 4)
    rng = MersenneTwister(42) 
    subset_indices = sample(rng, 1:total_test_samples, val_subset_size; replace=false)
    
    num_samples = size(train_images, 4);
    num_batches =  ceil(Int, num_samples / batch_size);

    init_learning_rate = 1.0e-4
    min_learning_rate = 1.0e-6
  
    drift_losses = Float32[]
    test_drift_losses = Float32[]
    validation_rmse = Float32[]

    best_test_loss_drift = Inf;
    patience = 25;
    counter = 0;
    stop_training = false;

    for epoch in 1:num_epochs
        if !stop_training 
            println("Epoch $epoch")
            shuffled_indices = randperm(size(train_images, 4))
            initial_train_images = initial_train_images[:, :, :, shuffled_indices]
            train_images = train_images[:, :, :, shuffled_indices]
            train_labels_closure = train_labels_closure[:, :, :, shuffled_indices]
            train_labels_state = train_labels_state[:, :, :, shuffled_indices]
        
            new_learning_rate = min_learning_rate .+ 0.5f0 .* (init_learning_rate - min_learning_rate) .* (1 .+ cos.(epoch ./ num_epochs .* π))
            Optimisers.adjust!(opt_drift, new_learning_rate)

            epoch_drift_loss = 0.0

            for batch_index in 1:num_batches-1
                initial_sample = Float32.(get_minibatch_NS(initial_train_images, batch_size, batch_index)) |> dev
                target_sample = Float32.(get_minibatch_NS(train_images, batch_size, batch_index)) |> dev  
                target_labels_closure_sample = Float32.(get_minibatch_NS(train_labels_closure, batch_size, batch_index)) |> dev 
                target_labels_state_sample = Float32.(get_minibatch_NS(train_labels_state, batch_size, batch_index)) |> dev 
                t_sample = Float32.(reshape(rand(Float32, batch_size), 1, 1, 1, batch_size)) |> dev
                z_sample = Float32.(randn(size(target_sample))) |> dev
                W_sample = sqrt.(t_sample) .* z_sample |> dev
                I_sample = Float32.(stochastic_interpolant(initial_sample, target_sample, W_sample, t_sample, 0.1)) |> dev

                loss_drift_closure = (ps_) -> begin
                    dI_dt_sample = Float32.(time_derivative_stochastic_interpolant(initial_sample, target_sample, W_sample, t_sample, 0.1))
                    velocity, st_drift = Lux.apply(velocity_cnn, (I_sample, t_sample, target_labels_closure_sample, target_labels_state_sample), ps_, st_drift)
                    return loss_fn(velocity, dI_dt_sample), st_drift
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
            num_test_samples = size(target_test, 4)
            num_test_batches = div(num_test_samples, batch_size) + (num_test_samples % batch_size > 0)

            _st_drift = Lux.testmode(st_drift) |> dev

            for test_batch_index in 1:num_test_batches
                test_initial_sample = Float32.(get_minibatch_NS(initial_test, batch_size, test_batch_index)) |> dev
                test_target_sample = Float32.(get_minibatch_NS(target_test, batch_size, test_batch_index)) |> dev 
                test_target_label_closure_sample = Float32.(get_minibatch_NS(target_label_closure_test, batch_size, test_batch_index)) |> dev
                test_target_label_state_sample = Float32.(get_minibatch_NS(target_label_state_test, batch_size, test_batch_index)) |> dev
                test_t_sample = Float32.(reshape(rand(Float32, batch_size), 1, 1, 1, batch_size)) |> dev
                test_z_sample = Float32.(randn(size(test_target_sample))) |> dev
                test_W_sample = sqrt.(test_t_sample) .* test_z_sample |> dev
                test_I_sample = Float32.(stochastic_interpolant(test_initial_sample, test_target_sample, test_W_sample, test_t_sample, 0.1)) |> dev
                test_dI_dt_sample = Float32.(time_derivative_stochastic_interpolant(test_initial_sample, test_target_sample, test_W_sample, test_t_sample, 0.1)) |> dev
                test_velocity, _ = Lux.apply(velocity_cnn, (test_I_sample, test_t_sample, test_target_label_closure_sample, test_target_label_state_sample), ps_drift, _st_drift)
                test_drift_loss += loss_fn(test_velocity, test_dI_dt_sample)
            end
            test_drift_loss /= num_test_batches
            push!(test_drift_losses, test_drift_loss)
            println("Validation loss for drift term: $test_drift_loss")

            if epoch % eval_frequency == 0
                y_pred = generate_closure(velocity_cnn, ps_drift, _st_drift, target_label_closure_test[:,:,:,subset_indices], target_label_state_test[:,:,:,subset_indices], initial_test[:,:,:,subset_indices], num_val_steps, dev; method=:heun)
                rmse = mean(mean_relative_mse(target_label_closure_test[:,:,:, (subset_indices .+ 1)], y_pred; dev))
                push!(validation_rmse, rmse)
                println("Validation RMSE (Epoch $epoch): ", rmse)
            end

            save_training_state(save_path, model_name, drift_losses, test_drift_losses, validation_rmse, num_finished_epochs)

            if test_drift_loss < best_test_loss_drift
                best_test_loss_drift = test_drift_loss 
                counter = 0  
                save_model("$save_path/$model_name.bson", ps_drift, st_drift, opt_drift; dev)
                println(" Saved intermediate model parameters ")
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

    p1 = plot(1:num_finished_epochs, drift_losses, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss", yscale=:log10)
    plot!(p1, 1:num_finished_epochs, test_drift_losses, label="Validation Loss")
    p2 = plot(1:eval_frequency:num_finished_epochs, validation_rmse, label= " Validation RMSE", xlabel="Epoch", ylabel="RMSE", title="Validation Accuracy (RMSE)")
    final_plot = plot(p1, p2, layout=(2,1), size=(800,800))

    savefig(final_plot, "figures/final_loss_plot_$(model_name).png")
    println("Final loss and accuracy plot saved at $(save_path)/final_loss_plot_$(model_name).png")

    println("Training completed")
    return ps_drift, st_drift, opt_drift
end

function save_training_state(save_path, model_name, drift_losses, test_drift_losses, validation_rmse, num_finished_epochs)
    training_state = Dict(
        :drift_losses => drift_losses,
        :test_drift_losses => test_drift_losses,
        :validation_rmse => validation_rmse,
        :num_finished_epochs => num_finished_epochs
    )

    BSON.@save "$(save_path)/$(model_name)_training_state.bson" training_state
    println("Training state saved at $(save_path)/$(model_name)_training_state.bson")
end
