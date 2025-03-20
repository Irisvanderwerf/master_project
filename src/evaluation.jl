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


function time_steps_dependency(test_data, ϵ; dev, method=:euler_maruyama, error_threshold=1e-6, improvement_threshold=1e-3)
    num_steps_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    num_time_steps = 100
    errors_over_time = Dict{Int, Vector{Float64}}()  
    for num_steps in num_steps_values
        errors_over_time[num_steps] = zeros(num_time_steps)
        initial_test_image = test_data.initial[:,:,:,1,1] |> dev 
        test_target = test_data.target[:,:,:,1,1] |> dev
        current_closure = initial_test_image 
        for t in 1:num_time_steps
            if method == :euler_maruyama
                pred_closure = generate_closure_SI(test_target, current_closure, num_steps, ϵ, dev; method=:euler_maruyama)
            elseif method == :heuns_method
                pred_closure = generate_closure_SI(test_target, current_closure, num_steps, ϵ, dev; method=:heuns_method)
            end
            pred_closure = mean(pred_closure, dims=4)[:,:,:,1]
            loss = prior(test_target[:,:,:,1], pred_closure; dev) |> dev
            errors_over_time[num_steps][t] = loss
            current_closure = pred_closure
            test_target = test_data.target[:,:,:,1+t, 1] |> dev
            if loss ≤ error_threshold
                println("Stopping at N=$num_steps because error threshold ($error_threshold) reached at time step $t.")
                break
            end
        end 
        mean_error = mean(errors_over_time[num_steps])
        println("N=$num_steps: Average Error over the Trajectory = $mean_error")
    end
    plt = plot(
    xlabel="Time Steps", 
    ylabel="A-priori error", 
    title=(method == :euler_maruyama ? "Euler-Maruyama" : "Heun"),
    lw=2,
    guidefontsize=14,  
    tickfontsize=12,    
    legendfontsize=12, 
    titlefontsize=16   
    )
    for num_steps in sort(collect(keys(errors_over_time)))
        plot!(plt, 1:num_time_steps, errors_over_time[num_steps], label="N=$num_steps")
    end
    ylims!(0,0.0006)
    savefig(plt, "error_evolution_$method.png")
    return errors_over_time
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