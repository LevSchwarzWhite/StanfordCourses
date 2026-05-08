from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    x_normalized = x.reshape(x.shape[0], -1)
    out = x_normalized.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    x_normalized = x.reshape(x.shape[0], -1)
    dw = x_normalized.T.dot(dout)
    db = np.sum(dout, axis=0)
    dx_normalized = dout.dot(w.T)
    dx = dx_normalized.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    
    N = x.shape[0]
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs[range(N), y])) / N
    dx = probs.copy()
    dx[range(N), y] -= 1
    dx /= N
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps) 
        out = gamma * x_normalized + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
    elif mode == 'test':
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        cache = None
        out = gamma * x_normalized + beta
    else:
        return ValueError("Invalid forward batchnorm mode '%s'" % mode)
    
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_normalized * dout, axis=0)
    dx_normalized = gamma * dout
    dvar = np.sum(dx_normalized * (x - sample_mean) * -0.5 * (sample_var + eps) ** (-1.5), axis=0)
    dmean = np.sum(dx_normalized * -1 / np.sqrt(sample_var + eps), axis=0) + dvar * np.sum(-2 * (x - sample_mean), axis=0) / N
    dx = dx_normalized / np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / N + dmean / N
    
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_normalized * dout, axis=0)
    
    std_inv = 1. / np.sqrt(sample_var + eps)
    
    dx = (gamma * std_inv / N) * (N * dout - np.sum(dout, axis=0) - x_normalized * np.sum(dout * x_normalized, axis=0))

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_normalized + beta
    cache = (x, x_normalized, gamma, beta, mean, var, eps)
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, x_normalized, gamma, beta, mean, var, eps = cache
    
    N, D = dout.shape
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    dhatx = dout * gamma
    
    std_inv = 1.0 / np.sqrt(var + eps)
    
    term1 = D * dhatx
    term2 = np.sum(dhatx, axis=1, keepdims=True)
    term3 = x_normalized * np.sum(dhatx * x_normalized, axis=1, keepdims=True)
    
    dx = (std_inv / D) * (term1 - term2 - term3)
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None
    
    N, D = x.shape

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        
    elif mode == "test":
        out = x
        mask = None

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    out = np.zeros((N, F, 1 + (H + 2 * pad - HH) // stride, 1 + (W + 2 * pad - WW) // stride))
    for f in range(F):
        for i in range(N):
            for j in range(0, H + 2 * pad - HH + 1, stride):
                for k in range(0, W + 2 * pad - WW + 1, stride):
                    x_padded = np.pad(x[i], ((0,0), (pad, pad), (pad, pad)), mode='constant')
                    out[i, f, j // stride, k // stride] = np.sum(x_padded[:, j:j+HH, k:k+WW] * w[f]) + b[f]
    
    
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = np.zeros_like(cache[0]), np.zeros_like(cache[1]), np.zeros_like(cache[2])
    x, w, b, conv_param = cache
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for f in range(F):
        for i in range(N):
            for j in range(0, H + 2 * pad - HH + 1, stride):
                for k in range(0, W + 2 * pad - WW + 1, stride):
                    x_padded = np.pad(x[i], ((0, 0), (pad, pad), (pad, pad)), mode='constant')
                    dx_padded = np.zeros_like(x_padded)
                    dx_padded[:, j:j+HH, k:k+WW] += w[f] * dout[i, f, j // stride, k // stride]
                    dw[f] += x_padded[:, j:j+HH, k:k+WW] * dout[i, f, j // stride, k // stride]
                    db[f] += dout[i, f, j // stride, k // stride]
                    dx[i] += dx_padded[:, pad:pad+H, pad:pad+W]
                    
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    out = np.zeros((N, C, 1 + (H - pool_height) // stride, 1 + (W - pool_width) // stride))
    for i in range(N):
        for j in range(C):
            for k in range(0, H - pool_height + 1, stride):
                for l in range(0, W - pool_width + 1, stride):
                    out[i, j, k // stride, l // stride] = np.max(x[i, j, k:k+pool_height, l:l+pool_width])
     
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    dx = np.zeros_like(x)
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    for i in range(N):
        for j in range(C):
            for k in range(0, H - pool_height + 1, stride):
                for l in range(0, W - pool_width + 1, stride):
                    x_pool = x[i, j, k:k+pool_height, l:l+pool_width]
                    max_val = np.max(x_pool)
                    mask = (x_pool == max_val)
                    dx[i, j, k:k+pool_height, l:l+pool_width] += mask * dout[i, j, k // stride, l // stride]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    
    N, C, H, W = dout.shape
    
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    
    N, C, H, W = x.shape
    
    x_grouped = x.reshape(N, G, C // G, H, W)
    
    mean = np.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)
    
    x_normalized_grouped = (x_grouped - mean) / np.sqrt(var + eps)
    
    x_normalized = x_normalized_grouped.reshape(N, C, H, W)
    
    out = x_normalized * gamma + beta
    
    cache = (x_grouped, x_normalized, gamma, beta, G, mean, var, eps)
    
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None
    
    N, C, H, W = dout.shape
    
    x_grouped, x_normalized, gamma, beta, G, mean, var, eps = cache
    
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3), keepdims=True)
    
    dx_normalized = dout * gamma
    
    dx_normalized_grouped = dx_normalized.reshape(N, G, C // G, H, W)
    
    M = (C // G) * H * W
    std_inv = 1. / np.sqrt(var + eps)
    
    sum_dx_norm = np.sum(dx_normalized_grouped, axis=(2, 3, 4), keepdims=True)
    
    sum_dx_norm_dot_x_sum = np.sum(dx_normalized_grouped * x_normalized.reshape(N, G, C // G, H, W), 
                                   axis=(2, 3, 4), keepdims=True)
    
    x_norm_grouped = x_normalized.reshape(N, G, C // G, H, W)
    
    dx_grouped = std_inv / M * (M * dx_normalized_grouped - sum_dx_norm - x_norm_grouped * sum_dx_norm_dot_x_sum)
    
    dx = dx_grouped.reshape(N, C, H, W)
    
    return dx, dgamma, dbeta
