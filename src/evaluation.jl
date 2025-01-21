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

function compute_metrics_average(y_true, y_pred, model_name; dev)
    y_true = y_true |> dev
    y_pred = y_pred |> dev

    num_samples = size(y_true, 4)

    mse = Float32[]              
    rrmse = Float32[]    

    for i in 1:num_samples
        y = y_true[:, :, :, i]
        y_hat = y_pred[:, :, :, i]
        push!(mse, mean_squared_error(y, y_hat; dev))
        push!(rrmse, relative_root_mse(y, y_hat; dev))
    end
    avg_mse = mean(mse)
    avg_rrmse = mean(rrmse)

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

function time_steps_dependency(test_data, velocity_cnn, ps_drift, _st_drift, ϵ, model_name; dev, method=:euler_maruyama)
    num_steps_values = [10, 20, 50, 100, 150, 200, 250, 300];
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
                avg_mse, avg_rrmse = compute_metrics_average(test_images, pred_closure, model_name; dev)
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