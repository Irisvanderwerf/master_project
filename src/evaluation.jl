function mean_squared_error(y, y_pred; dev)
    y = y |> dev
    y_pred = y_pred |> dev
    mse = mean((y .- y_pred).^2)
    return mse
end

function mean_relative_mse(y, y_pred; dev)
    y = y |> dev
    y_pred = y_pred |> dev
    mse = mean_squared_error(y, y_pred; dev)
    variance_y = mean((y .- mean(y)).^2)
    rel_mse = mse / variance_y
    return rel_mse
end

function relative_rmse(y, y_pred; dev)
    y = y |> dev
    y_pred = y_pred |> dev
    rel_mse = mean_relative_mse(y, y_pred; dev)
    rel_rmse = sqrt(rel_mse)
    return rel_rmse
end

function compute_metrics_average(y_true, y_pred, model_name; dev)
    y_true = y_true |> dev
    y_pred = y_pred |> dev

    num_samples = size(y_true, 4)

    mse = Float32[]         
    rel_mse = Float32[]     
    rel_rmse = Float32[]    

    for i in 1:num_samples
        y = y_true[:, :, :, i]
        y_hat = y_pred[:, :, :, i]
        push!(mse, mean_squared_error(y, y_hat; dev))
        push!(rel_mse, mean_relative_mse(y, y_hat; dev))
        push!(rel_rmse, relative_rmse(y, y_hat; dev))
    end
    avg_mse = mean(mse)
    avg_rel_mse = mean(rel_mse)
    avg_rel_rmse = mean(rel_rmse)

    println("The model $model_name achieved the following accuracy:")
    println("Average Mean Squared Error (MSE): $avg_mse")
    println("Average Relative MSE: $avg_rel_mse")
    println("Average Relative RMSE: $avg_rel_rmse")

    return avg_mse, avg_rel_mse, avg_rel_rmse
end