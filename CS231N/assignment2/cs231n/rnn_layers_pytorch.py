"""This file defines layer types that are commonly used for recurrent neural networks.
"""
import torch


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A torch array containing input data, of shape (N, d_1, ..., d_k)
    - w: A torch array of weights, of shape (D, M)
    - b: A torch array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    out = x.reshape(x.shape[0], -1) @ w + b
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    """
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)
    return next_h


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    """
    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros((N, T, H), device=x.device, dtype=x.dtype)
    prev_h = h0
    for t in range(T):
        h_t = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = h_t
        prev_h = h_t
    return h


def word_embedding_forward(x, W):
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """
    out = W[x]
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    """
    x = torch.from_numpy(x)
    prev_h = torch.from_numpy(prev_h)
    prev_c = torch.from_numpy(prev_c)
    Wx = torch.from_numpy(Wx)
    Wh = torch.from_numpy(Wh)
    b = torch.from_numpy(b)
    
    next_h, next_c = None, None
    
    a = x @ Wx + prev_h @ Wh + b
    ai, af, ao, ag = torch.split(a, a.shape[1] // 4, dim=1)
    i, f, o, g = torch.sigmoid(ai), torch.sigmoid(af), torch.sigmoid(ao), torch.tanh(ag)
    next_c = f * prev_c + i * g
    next_h = o * torch.tanh(next_c)
    cache = (x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c)
    
    next_h = next_h.numpy()
    next_c = next_c.numpy()

    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    "lstm was removed from the 2026 assignment, so there is no documentation for this function."
    x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_c = cache
    
    inputs = (x, prev_h, prev_c, Wx, Wh, b)
    for tensor in inputs:
        tensor.requires_grad_(True)
        
    a = x @ Wx + prev_h @ Wh + b
    ai, af, ao, ag = torch.split(a, a.shape[1] // 4, dim=1)
    i, f, o, g = torch.sigmoid(ai), torch.sigmoid(af), torch.sigmoid(ao), torch.tanh(ag)
    next_c = f * prev_c + i * g
    next_h = o * torch.tanh(next_c)
    
    grad_h = torch.from_numpy(dnext_h)
    grad_c = torch.from_numpy(dnext_c)
    
    loss = (next_h * grad_h).sum() + (next_c * grad_c).sum()
    loss.backward()
    
    dx = x.grad.numpy()
    dh = prev_h.grad.numpy()
    dc = prev_c.grad.numpy()
    dWx = Wx.grad.numpy()
    dWh = Wh.grad.numpy()
    db = b.grad.numpy()
    
    return dx, dh, dc, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """
    x = torch.from_numpy(x)
    h0 = torch.from_numpy(h0)
    Wx = torch.from_numpy(Wx)
    Wh = torch.from_numpy(Wh)
    b = torch.from_numpy(b)
    
    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros((N, T, H), device=x.device, dtype=x.dtype)
    prev_h = h0
    prev_c = torch.zeros_like(h0)	
    for t in range(T):
        h_t, c_t, _ = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        h[:, t, :] = h_t
        prev_h = h_t
        prev_c = c_t
        
    cache = (x, h0, Wx, Wh, b, h)
    return h, cache

def lstm_backward(dout, cache):
	"lstm was removed from the 2026 assignment, so there is no documentation for this function."
 
	x, h0, Wx, Wh, b, h = cache
	N, T, D = x.shape
	_, H = h.shape
	
	x_t = torch.from_numpy(x).requires_grad_(True)
	prev_h_t = torch.from_numpy(h).requires_grad_(True)
	Wx_t = torch.from_numpy(Wx).requires_grad_(True)
	Wh_t = torch.from_numpy(Wh).requires_grad_(True)
	b_t = torch.from_numpy(b).requires_grad_(True)
 
	h_list = []
	h_cur = prev_h_t
	c_cur = torch.zeros_like(h_cur, devaice=x.device, dtype=x.dtype)
 
	for t in range(T):
		h_cur, c_cur, _ = lstm_step_forward(x_t[:, t, :], h_cur, c_cur, Wx_t, Wh_t, b_t)
		h_list.append(h_cur)
  
	next_h_t = torch.stack(h_list, dim=1)
 
	external_grad = torch.from_numpy(dout)
	next_h_t.backward(external_grad)
 
	dx = x_t.grad.numpy()
	dh = prev_h_t.grad.numpy()
	dWx = Wx_t.grad.numpy()
	dWh = Wh_t.grad.numpy()
	db = b_t.grad.numpy()
 
	return dx, dh, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
