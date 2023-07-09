# Import libraries
using LinearAlgebra
using Random

function sigmoid(x)
    """
    Computes the sigmoid activation function element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor with the sigmoid activation applied element-wise.
    """
    return 1.0 ./ (1.0 .+ exp.(-x))
end


function gru_cell_forward(x, prev_h, parameters)
    """
    Implements one GRU cell in an unrolled loop and returns its hidden output alongside other values

    Args:
        x: Input tensor at the current time step, shape (input_size, n_samples).
        prev_h: Hidden state tensor from the previous time step, shape (hidden_size, n_samples).
        parameters: Dictionary containing:
            Wz: Weight matrix for the update gate, shape (hidden_size, hidden_size + input_size).
            Wr: Weight matrix for the reset gate, shape (hidden_size, hidden_size + input_size).
            Wh: Weight matrix for the new memory cell, shape (hidden_size, hidden_size + input_size).
            Wy: Weight matrix for the hidden state of the output, shape (output_size, hidden_size).
            bz: Bias vector for the update gate, shape (hidden_size, 1).
            br: Bias vector for the reset gate, shape (hidden_size, 1).
            bh: Bias vector for the new memory cell, shape (hidden_size, 1).
            by: Bias vector for the hidden state to the output, shape (output_size, 1)
    Returns:
        next_h: Next hidden state tensor, shape (hidden_size, n_samples).
        y_pred: prediction at timestep "t", numpy array of shape, shape (output_size, n_samples)
        cache: dictionary to be passed into `gru_cell_backward` function later on.
    """
    # Retrieve parameters from "parameters"
    Wz = parameters["Wz"]
    Wr = parameters["Wr"]
    Wh = parameters["Wh"]
    Wy = parameters["Wy"]
    bz = parameters["bz"]
    br = parameters["br"]
    bh = parameters["bh"]
    by = parameters["by"]

    # Concatenate prev_h and x
    concat = vcat(prev_h, x)

    # Compute values for zt, htilde_t, h_next
    z = sigmoid(Wz * concat .+ bz)         # Update gate
    r = sigmoid(Wr * concat .+ br)          # Reset gate
    h_tilde = tanh.(Wh * vcat(r .* prev_h, x) .+ bh)       # New memory cell
    next_h = (1.0 .- z) .* prev_h + z .* h_tilde    # Hidden state at current time step

    # Compute prediction of the GRU cell (e.g., softmax)
    y_pred = Wy * next_h .+ by

    # store values needed for backward propagation in cache
    cache = (next_h, prev_h, z, r, h_tilde, x, parameters)

    return next_h, y_pred, cache
end


function gru_forward(x, ho, parameters)
    """
    Performs the forward pass of a Gated Recurrent Unit (GRU).

    Args:
        x: Input tensor for all the time steps, shape (input_size, n_samples, total_grus).
        ho: Initial hidden state tensor, shape (hidden_size, n_samples).
        parameters: Dictionary containing:
            Wz: Weight matrix for the update gate, shape (hidden_size, hidden_size + input_size).
            Wr: Weight matrix for the reset gate, shape (hidden_size, hidden_size + input_size).
            Wh: Weight matrix for the new memory cell, shape (hidden_size, hidden_size + input_size).
            Wy: Weight matrix for the hidden state of the output, shape (output_size, hidden_size).
            bz: Bias vector for the update gate, shape (hidden_size, 1).
            br: Bias vector for the reset gate, shape (hidden_size, 1).
            bh: Bias vector for the new memory cell, shape (hidden_size, 1).
            by: Bias vector for the hidden state to the output, shape (output_size, 1)
    Returns:
        h: Hidden state tensor for all the time steps (hidden_size, n_samples, total_grus).
        y: Predictions for every time-step, numpy array of shape (output_size, n_samples, total_grus)
        caches: List of all caches for the backward pass (including x).
    """
    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x
    Wy = parameters["Wy"]
    input_size, n_samples, total_grus = size(x)
    output_size, hidden_size = size(Wy)

    # initialize "h" and "next_h" with zeros
    h = zeros(hidden_size, n_samples, total_grus)
    y = zeros(output_size, n_samples, total_grus)
    next_h = ho

    # loop over all time-steps
    for t in 1:total_grus
        next_h, y_pred, cache = gru_cell_forward(x[:, :, t], next_h, parameters)
        y[:, :, t] = y_pred
        h[:, :, t] = next_h
        push!(caches, cache)
    end

    caches = (caches, x)

    return h, y, caches
end


function gru_cell_backward(dh, cache)
    """
    Performs the backward pass for a single GRU cell.

    Args:
        dh: Gradient of the hidden state, shape, shape (hidden_size, n_samples).
        cache: Cache storing the information of the forward pass.

    Returns:
        gradients: Dictionary containing:
            dx: Gradient of the input tensor x, shape (input_size, n_samples).
            dh_prev: Gradient of the previous hidden state, shape (hidden_size, n_samples).
            dWz: Gradient of the weight matrix Wz, shape (hidden_size, hidden_size + input_size).
            dWr: Gradient of the weight matrix Wr, shape (hidden_size, hidden_size + input_size).
            dWh: Gradient of the weight matrix Wh, shape (hidden_size, hidden_size + input_size).
            dbz: Gradient of the bias vector bz, shape (hidden_size, 1).
            dbr: Gradient of the bias vector br, shape (hidden_size, 1).
            dbh: Gradient of the bias vector bh, shape (hidden_size, 1).
    """
    # Retrieve information from "cache"
    (h, prev_h, z, r, h_tilde, x, parameters) = cache

    # Retrieve parameters from "parameters"
    Wz = parameters["Wz"]
    Wr = parameters["Wr"]
    Wh = parameters["Wh"]
    bz = parameters["bz"]
    br = parameters["br"]
    bh = parameters["bh"]

    # Checking the input
    input_size = size(x, 1)

    # Compute gates related derivatives
    dh_tilde = dh .* z .* (1.0 .- tanh.(h_tilde) .^ 2)  # Gradient of the new memory cell
    dz = (dh .* h_tilde .+ 1.0 .- dh .* prev_h) .* (sigmoid.(z) .* (1.0 .- sigmoid.(z)))  # Gradient of the update gate
    dr = Wh[:, 1:end-input_size]' * dh_tilde .* r .* sigmoid.(r) .* (1.0 .- sigmoid.(r))  # Gradient of the reset gate
    dh = Wh[:, 1:end-input_size]' * dh_tilde .* r + (1.0 .- z) .* dh + Wr[:, 1:end-input_size]' * dr + Wz[:, 1:end-input_size]' * dz  # Gradient of the input to the new memory cell
    dx = Wh[:, end-input_size+1:end]' * dh_tilde + Wz[:, end-input_size+1:end]' * dz + Wr[:, end-input_size+1:end]' * dr  # Gradient of the input tensor x

    # Concatenate prev_h and x
    concat = vcat(prev_h, x)

    # Compute parameters related derivatives
    dWz = dz * concat'  # Gradient of the weight matrix Wz
    dWr = dr * concat'  # Gradient of the weight matrix Wr
    dWh = dh * vcat(prev_h .* r, x)'  # Gradient of the weight matrix Wh
    dbz = sum(dz, dims=2)  # Gradient of the bias vector bz
    dbr = sum(dr, dims=2)  # Gradient of the bias vector br
    dbh = sum(dh, dims=2)  # Gradient of the bias vector bh

    dh_prev = dh  # Gradient of the previous hidden state

    # Save gradients in dictionary
    gradients = Dict("dx" => dx, "dh_prev" => dh_prev, "dWz" => dWz,
                    "dWr" => dWr, "dWh" => dWh, "dbz" => dbz,
                    "dbr" => dbr, "dbh" => dbh, "dh" => dh)

    return gradients
end


function gru_backward(dh, caches)
    """
    Performs the backward pass of a Gated Recurrent Unit (GRU).

    Args:
        dh: Gradient of the hidden state tensor from the subsequent layers, shape (hidden_size, n_samples, total_grus).
        caches: Caches from the forward pass, which include the information needed for backpropagation.

    Returns:
        gradients: Dictionary containing the gradients of the parameters and inputs.
            - dx: Gradient of the input tensor x, shape (input_size, n_samples, total_grus).
            - dh_prev: Gradient of the previous hidden state tensor, shape (hidden_size, n_samples, total_grus).
            - dWz: Gradient of the weight matrix Wz, shape (hidden_size, hidden_size + input_size).
            - dWr: Gradient of the weight matrix Wr, shape (hidden_size, hidden_size + input_size).
            - dWh: Gradient of the weight matrix Wh, shape (hidden_size, hidden_size + input_size).
            - dbz: Gradient of the bias vector bz, shape (hidden_size, 1).
            - dbr: Gradient of the bias vector br, shape (hidden_size, 1).
            - dbh: Gradient of the bias vector bh, shape (hidden_size, 1).
    """
    # Retrieve values from the first cache
    (caches, x) = caches
    (h1, ho, z1, r1, h_tilde1, x1, parameters) = caches[1]

    # Retrieve dimensions from dh's and x1's shapes
    hidden_size, n_samples, total_grus = size(dh)
    input_size = size(x1, 1)

    # initialize the gradients with the right sizes
    dx = zeros(input_size, n_samples, total_grus)
    dh_prev = zeros(hidden_size, n_samples, total_grus)
    dWz = zeros(hidden_size, hidden_size + input_size)
    dWr = zeros(hidden_size, hidden_size + input_size)
    dWh = zeros(hidden_size, hidden_size + input_size)
    dbz = zeros(hidden_size, 1)
    dbr = zeros(hidden_size, 1)
    dbh = zeros(hidden_size, 1)

    # loop over all time-steps
    for t in total_grus:-1:1
        gradients = gru_cell_backward(dh[:, :, t], caches[t])
        dx[:,:,t] = gradients["dx"]
        dWz += gradients["dWz"]
        dWr += gradients["dWr"]
        dWh += gradients["dWh"]
        dbz += gradients["dbz"]
        dbr += gradients["dbr"]
        dbh += gradients["dbh"]
        dh_prev[:,:,t] = gradients["dh_prev"]
    end

    # Save gradients in dictionary
    gradients = Dict("dx" => dx, "dh_prev" => dh_prev, "dWz" => dWz,
                    "dWr" => dWr, "dWh" => dWh, "dbz" => dbz,
                    "dbr" => dbr, "dbh" => dbh)

    return gradients
end


function train_gru(x_train, y_train, hidden_size, learning_rate, num_epochs, checking=true)
  """
  Trains a Gated Recurrent Unit (GRU) model using the provided training data.

  Args:
      x_train: Input training data tensor, shape (input_size, n_samples, total_grus).
      y_train: Target training data tensor, shape (output_size, n_samples, total_grus).
      hidden_size: Dimensionality of the hidden state
      num_epochs: Number of training epochs.
      learning_rate: Learning rate for gradient descent.
      checking: Whether looking for the best cost.
  Returns:
      Trained parameters of the GRU model: Wz, Wr, Wh, bz, br, bh.
  """
    # Initialize parameters
    input_size, n_samples, total_grus = size(x_train)
    output_size = size(y_train, 1)
    Random.seed!(123)
    Wz = randn(hidden_size, hidden_size + input_size)
    Wr = randn(hidden_size, hidden_size + input_size)
    Wh = randn(hidden_size, hidden_size + input_size)
    Wy = randn(output_size, hidden_size)
    bz = randn(hidden_size, 1)
    br = randn(hidden_size, 1)
    bh = randn(hidden_size, 1)
    by = randn(output_size, 1)
    ho = zeros(hidden_size, n_samples)
    dh = zeros(hidden_size, n_samples, total_grus)

    # Generate a dictionary of parameters
    parameters = Dict("Wz" => Wz, "Wr" => Wr, "Wh" => Wh, "Wy" => Wy,
                    "bz" => bz, "br" => br, "bh" => bh, "by" => by)
  
    # Training loop
    cost_per_epoch = []
    best_cost = 1000000
    final_params = deepcopy(parameters)
    for epoch in 1:num_epochs

        # Forward pass
        h, y_pred, caches = gru_forward(x_train, ho, parameters)

        # Compute cost (e.g., mean squared error)
        cost = sum((y_pred - y_train).^2) / (n_samples * total_grus * output_size)
        push!(cost_per_epoch, cost)

        # Backward pass
        gradients = gru_backward(dh, caches)

        # Parameter updates using gradient descent
        Wz -= learning_rate .* gradients["dWz"]
        Wr -= learning_rate .* gradients["dWr"]
        Wh -= learning_rate .* gradients["dWh"]
        bz -= learning_rate .* gradients["dbz"]
        br -= learning_rate .* gradients["dbr"]
        bh -= learning_rate .* gradients["dbh"]
        dh -= learning_rate .* gradients["dh_prev"]
        parameters = Dict("Wz" => Wz, "Wr" => Wr, "Wh" => Wh, "Wy" => Wy,
                    "bz" => bz, "br" => br, "bh" => bh, "by" => by)

        # Print loss for each epoch
        println("Epoch: $epoch, Cost: $cost")

        # Checking best cost
        if checking
            if  cost < best_cost
                final_params = Dict("Wz" => Wz, "Wr" => Wr, "Wh" => Wh, "Wy" => Wy,
                        "bz" => bz, "br" => br, "bh" => bh, "by" => by)
            else
                final_params = deepcopy(parameters)
            end
            best_cost = cost
        else
            final_params = deepcopy(parameters)
        end
    end

    return final_params, cost_per_epoch
end


function predict_gru(x, parameters)
    """
    Make predictions using the trained GRU model on the test data.

    Args:
        x: Input testing data.
        parameters: Dictionary containing:
            Wz: Weight matrix for the update gate.
            Wr: Weight matrix for the reset gate.
            Wh: Weight matrix for the new memory cell.
            Wy: Weight matrix for the hidden state of the output.
            bz: Bias vector for the update gate.
            br: Bias vector for the reset gate.
            bh: Bias vector for the new memory cell.
            by: Bias vector for the hidden state to the output.

    Returns:
        predictions: List of predictions.
    """
    hidden_size = size(parameters["Wz"], 1)
    n_samples = size(x,2)
    ho = zeros(hidden_size, n_samples)
    _, predictions, _  = gru_forward(x, ho, parameters)
    return predictions
end


