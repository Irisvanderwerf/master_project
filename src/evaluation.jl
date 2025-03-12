function mean_squared_error(y, y_pred; dev)
    y = y |> dev
    y_pred = y_pred |> dev
    mse = mean((y .- y_pred).^2)
    return mse
end

function relative_root_mse(y, y_pred; dev)
    mse = mean_squared_error(y, y_pred; dev)
    normalization_factor = sum(y_pred.^2) 
    rrmse = sqrt.(mse ./ normalization_factor)
    return rrmse
end

function prior(y, y_pred; dev)
    y = y |> dev
    y_pred = y_pred |> dev
    loss = sum(abs2, y_pred .- y) ./ sum(abs2, y)
    return loss
end

function compute_metrics_average(y_true, y_pred, model_name; dev)
    y_true = y_true |> dev
    y_pred = y_pred |> dev

    num_samples = size(y_true, 4)

    mse = Float32[]              
    rrmse = Float32[]    

    for i in 1:num_samples
        y = y_true[:, :, :, i]
        y_hat = y_pred[:, :, :, i]
        push!(prior_error, prior(y, y_hat; dev))
    end
    mean_prior = mean(prior_error)

    println("The model $model_name achieved the following accuracy:")
    println("Average Mean Squared Error (MSE): $avg_mse")
    println("Average Relative Root Mean Square Error (RRMSE): $avg_rrmse")

    return avg_mse, avg_rrmse
end

function plot_closure_prediction(test_data, time, trajectory, num_steps, velocity_cnn, ps_drift, _st_drift, ϵ; dev, method=:euler_maruyama)
    initial_test_images, test_images, test_labels_closure, test_labels_state = repeat(test_data.initial[:,:,:,time,trajectory], 1,1,1,10), repeat(test_data.target[:,:,:,time,trajectory],1,1,1,10), repeat(test_data.closure[:,:,:,time,trajectory],1,1,1,10), repeat(test_data.state[:,:,:,time,trajectory],1,1,1,10) |> dev;
    if method == :euler_maruyama
        pred_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, test_labels_closure, test_labels_state, initial_test_images, num_steps, ϵ, dev; method=:euler_maruyama)
    elseif method == :heuns_method
        pred_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, test_labels_closure, test_labels_state, initial_test_images, num_steps, ϵ, dev; method=:heuns_method)
    end
    pred_closure_mean = Array(mean(pred_closure, dims=4))
    true_closure_mean = Array(mean(test_images, dims=4))
    
    p1 = heatmap(pred_closure_mean[:, :, 1], title="x-component prediction", xlabel="X", ylabel="Y", color=:viridis)
    p2 = heatmap(pred_closure_mean[:, :, 2], title="y-component prediction", xlabel="X", ylabel="Y", color=:viridis)
    p3 = heatmap(true_closure_mean[:, :, 1], title="x-component ground truth", xlabel="X", ylabel="Y", color=:viridis)
    p4 = heatmap(true_closure_mean[:, :, 2], title="y-component ground truth", xlabel="X", ylabel="Y", color=:viridis)

    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800)) 
    savefig(p, "closure_components_$method.png") 
end

function quality_deterministic_model(test_data, velocity_cnn_det, ps_deterministic, _st_deterministic, setup_les, psolver_les, N_les; dev)
    prior_losses = Dict{Int, Vector{Float64}}()
    computation_times = Dict{Int, Vector{Float64}}()
    for i in 1:4
        prior_losses[i] = Float32[]
        computation_times[i] = Float32[]
            
        batch_closure_values = Float32[]
        total_time = 0.0  
        for j in 1:100
            test_labels_closure, test_target = repeat(test_data.state[:,:,:,j,i],1,1,1,1), repeat(test_data.target[:,:,:,j,i],1,1,1,1) |> dev;
            elapsed_time = @elapsed begin
                pred_closure, _ = Lux.apply(velocity_cnn_det, (test_labels_closure), ps_deterministic, _st_deterministic) |> dev;
            end
            total_time += elapsed_time
            pad_pred_closure = pad_circular(pred_closure[:,:,:,1], 1; dims=1:2)   
            proj_pad_pred_closure = INS.project(pad_pred_closure, setup_les; psolver=psolver_les)
            proj_pred_closure = proj_pad_pred_closure[2:end-1, 2:end-1, :] |> dev
            loss = prior(test_target, proj_pred_closure; dev) |> dev
            push!(batch_closure_values, loss)
        end
        avg_time = total_time / 100
        push!(computation_times[i], avg_time)
        prior_loss = mean(batch_closure_values)
        push!(prior_losses[i], prior_loss)
        println("Trajectory $i: A-Priori Loss = $prior_loss, and computation time = $avg_time")
    end
end

function time_steps_dependency(test_data, velocity_cnn, ps_drift, _st_drift, ϵ, setup_les, psolver_les, N_les; dev, method=:euler_maruyama)
    num_steps_values = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    prior_losses = Dict{Int, Vector{Float64}}()
    computation_times = Dict{Int, Vector{Float64}}()
    for i in 1:4
        prior_losses[i] = Float32[]
        computation_times[i] = Float32[]
        for num_steps in num_steps_values
            batch_closure_values = Float32[]
            total_time = 0.0  
            for j in 1:100
                initial_test_images, test_labels_closure, test_labels_state, test_target = repeat(test_data.initial[:,:,:,j,i], 1,1,1,10), repeat(test_data.closure[:,:,:,j,i],1,1,1,10), repeat(test_data.state[:,:,:,j,i],1,1,1,10), repeat(test_data.target[:,:,:,j,i],1,1,1,10) |> dev;
                elapsed_time = @elapsed begin
                    if method == :euler_maruyama
                        pred_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, test_labels_closure, test_labels_state, initial_test_images, num_steps, ϵ, dev; method=:euler_maruyama)
                    elseif method == :heuns_method
                        pred_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, test_labels_closure, test_labels_state, initial_test_images, num_steps, ϵ, dev; method=:heuns_method)
                    end
                end
                total_time += elapsed_time
                pred_closure = mean(pred_closure, dims=4)[:,:,:,1]
                pad_pred_closure = pad_circular(pred_closure, 1; dims=1:2)   
                proj_pad_pred_closure = INS.project(pad_pred_closure, setup_les; psolver=psolver_les)
                proj_pred_closure = proj_pad_pred_closure[2:end-1, 2:end-1, :] |> dev
                loss = prior(test_target[:,:,:,1], proj_pred_closure; dev) |> dev
                push!(batch_closure_values, loss)
            end
            avg_time = total_time / 100
            push!(computation_times[i], avg_time)
            prior_loss = mean(batch_closure_values)
            push!(prior_losses[i], prior_loss)
            println("Trajectory $i, num_steps $num_steps: A-Priori Loss = $prior_loss, and computation time = $avg_time")
        end
    end

    plt = plot()
    for i in 1:4
        plot!(plt, num_steps_values, prior_losses[i], label="Trajectory $i", xlabel="num_steps", ylabel="A-priori error", lw=2)
    end
    if method == :euler_maruyama
        title!("Euler-Maruyama")
    elseif method == :heuns_method
        title!("Heun")
    end
    savefig(plt, "prior_error_$method.png")

    plt2 = plot()
    for i in 1:4
        plot!(plt2, num_steps_values, computation_times[i], label="Trajectory $i", xlabel="num_steps", ylabel="Computation Time (s)", lw=2)
    end
    if method == :euler_maruyama
        title!("Euler-Maruyama")
    elseif method == :heuns_method
        title!("Heun")
    end
    savefig(plt2, "computation_time_$method.png")
end

function time_steps_dependency(test_data, velocity_cnn, ps_drift, _st_drift, ϵ, model_name; dev, method=:euler_maruyama)
    num_steps_values = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    rmse_results = Dict{Int, Vector{Float64}}()
    for i in 1:4
        rmse_results[i] = Float32[]
        for num_steps in num_steps_values
            rmse_values = Float32[]
            for j in 1:100
                initial_test_images, test_images, test_labels_closure, test_labels_state = repeat(test_data.initial[:,:,:,j,i], 1,1,1,10), repeat(test_data.target[:,:,:,j,i],1,1,1,10), repeat(test_data.closure[:,:,:,j,i],1,1,1,10), repeat(test_data.state[:,:,:,j,i],1,1,1,10) |> dev;
                if method == :euler_maruyama
                    pred_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, test_labels_closure, test_labels_state, initial_test_images, num_steps, ϵ, dev; method=:euler_maruyama)
                elseif method == :heuns_method
                    pred_closure = generate_closure(velocity_cnn, ps_drift, _st_drift, test_labels_closure, test_labels_state, initial_test_images, num_steps, ϵ, dev; method=:heuns_method)
                end
                prior_loss = compute_metrics_average(test_images, pred_closure, model_name; dev)
                push!(rmse_values, avg_rrmse)
            end
            mean_rmse = mean(rmse_values)
            push!(rmse_results[i], mean_rmse)
            println("Trajectory $i, num_steps $num_steps: Mean RRMSE = $mean_rmse")
        end
    end

    plt = plot()
    for i in 1:4
        plot!(plt, num_steps_values, rmse_results[i], label="Trajectory $i", xlabel="num_steps", ylabel="Mean RRMSE", lw=2)
    end
    title!("Mean RRMSE vs num_steps for All Trajectories")
    savefig(plt, "rrmse_vs_num_steps_$method.png")
end