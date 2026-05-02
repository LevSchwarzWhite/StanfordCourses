from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + hidden_dims + [num_classes]
        
        for i in range(1, self.num_layers + 1):
            self.params[f"W{i}"] = weight_scale * np.random.randn(dims[i - 1], dims[i])
            self.params[f"b{i}"] = np.zeros(dims[i])
            
        if self.normalization in ["batchnorm", "layernorm"]:
            for i in range(1, self.num_layers):
                self.params[f"gamma{i}"] = np.ones(dims[i])
                self.params[f"beta{i}"] = np.zeros(dims[i])
        
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        
        caches = {}
        out = X
        
        for i in range(1, self.num_layers):
            Wi, bi = self.params[f"W{i}"], self.params[f"b{i}"]
            
            out, fc_cache = affine_forward(out, Wi, bi)
            
            bn_cache = None
            if self.normalization == "batchnorm":
                gamma, beta = self.params[f"gamma{i}"], self.params[f"beta{i}"]
                out, bn_cache = batchnorm_forward(out, gamma, beta, self.bn_params[i - 1])
            elif self.normalization == "layernorm":
                gamma, beta = self.params[f"gamma{i}"], self.params[f"beta{i}"]
                out, bn_cache = layernorm_forward(out, gamma, beta, self.bn_params[i - 1])
            
            out, relu_cache = relu_forward(out)
            
            do_cache = None
            if self.use_dropout:
                out, do_cache = dropout_forward(out, self.dropout_param)
                
            caches[i] = (fc_cache, bn_cache, relu_cache, do_cache)
            
        WL, bL = self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"]
        scores, cache_last = affine_forward(out, WL, bL)

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        
        for i in range(1, self.num_layers + 1):
            Wi = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(Wi * Wi)
            
        grads = {}
        
        dx, dWL, dbL = affine_backward(dscores, cache_last)
        grads[f"W{self.num_layers}"] = dWL + self.reg * WL
        grads[f"b{self.num_layers}"] = dbL
        
        for i in range(self.num_layers - 1, 0, -1):
            fc_cache, bn_cache, relu_cache, do_cache = caches[i]
            
            if self.use_dropout:
                dx = dropout_backward(dx, do_cache)
                
            dx = relu_backward(dx, relu_cache)
            
            if self.normalization == "batchnorm":
                if "batchnorm_backward" in globals():
                    dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache)
                else:
                    dx, dgamma, dbeta = batchnorm_backward_alt(dx, bn_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
            
            elif self.normalization == "layernorm":
                dx, dgamma, dbeta = layernorm_backward(dx, bn_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
                
            dx, dWi, dbi = affine_backward(dx, fc_cache)
            grads[f"W{i}"] = dWi + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = dbi

        return loss, grads
