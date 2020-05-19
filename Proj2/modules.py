"""Define all classes that relate to the Module class."""
import math

from torch import empty


class Module(object):
    """Generic module class."""

    def __init__(self):
        """Initialize the module."""
        self.name = "Undefined"
        self.type = "Module"
        self.data = None

    def forward(self, data):
        """Do a forward pass, MUST IMPLEMENT."""
        raise NotImplementedError

    def backward(self, upstream_derivative):
        """Do a backward step, MUST IMPLEMENT."""
        raise NotImplementedError

    @property
    def param(self):
        """Return all params, nothing by default."""
        return []

    def __repr__(self):
        """Print when called."""
        return "{} {}".format(self.type, self.name)

    def reset(self):
        """By default just re-initialize."""
        self.__init__()


class Linear(Module):
    """Define a Linear=Fully connected layer."""

    def __init__(self, dim_in, dim_out, w_init="normal"):
        """Initialize all metadata and weight/bias/gradients."""
        super().__init__()
        # metadata
        self.name = "Linear {}x{}".format(dim_in, dim_out)
        self.type = "Layer"
        self.weights_init = w_init
        self.dim_in = dim_in
        self.dim_out = dim_out

        # initialize bias
        self.bias = empty(dim_out).normal_()
        self.grad_bias = empty(dim_out).zero_()

        # initialize w
        self.weights = empty(dim_in, dim_out).normal_()
        if w_init.lower() == "he":
            self.weights.normal_().mul_(math.sqrt(2/(self.dim_in)))
        elif w_init.lower() == 'xavier':
            self.weights.normal_().mul_(
                math.sqrt(2/(self.dim_in + self.dim_out)))

        self.grad_weight = empty(dim_in, dim_out).zero_()

    def forward(self, data):
        """Do a forward pass."""
        self.data = data  # useful for backward step
        return self.data.matmul(self.weights).add(self.bias)  # Broadcast

    def backward(self, gradwrtoutput):
        """Do a backward pass."""
        self.grad_weight.add_(self.data.t().matmul(gradwrtoutput))
        self.grad_bias.add_(gradwrtoutput.sum(0))
        return gradwrtoutput.matmul(self.weights.t())

    def zero_grad(self):
        """Set all gradients to 0."""
        self.grad_weight.zero_()
        self.grad_bias.zero_()

    def reset(self):
        """Reset all weights and gradients as if newly initialized."""
        self.__init__(self.dim_in, self.dim_out, self.weights_init)

    @property
    def param(self):
        """Return all parameters."""
        return [(self.weights, self.grad_weight),
                (self.bias, self.grad_bias)]


class ReLU(Module):
    """Define the ReLU module."""

    def __init__(self):
        """Initialize metadata."""
        super().__init__()
        self.name = "ReLU"
        self.type = "Activation"
        self.positive_entries = None

    def forward(self, data):
        """Compute the forward pass."""
        self.data = data
        self.positive_entries = (data > 0).float()
        ret = self.data.mul(self.positive_entries)
        return ret

    def backward(self, gradwrtoutput):
        """Do a backward pass.

        Elementwise backward: if positive -> delta l / delta x
        otherwise 0 --> positive * delta elementwise
        """
        return self.positive_entries.mul(gradwrtoutput)


class Tanh(Module):
    """Define the Hyperbolic tangent module."""

    def __init__(self):
        """Initialize the module."""
        super().__init__()
        self.name = "Tanh"
        self.type = "Activation"
        self.dim_data = None
        self.tanh = None

    def forward(self, data):
        """Do a forward pass."""
        self.dim_data = data.shape
        self.tanh = data.tanh()
        return self.tanh

    def backward(self, gradwrtoutput):
        """Do a backward step.

        gradient * (1-tanh^2)
        """
        return gradwrtoutput * (1 - self.tanh.pow(2))


class Sigmoid(Module):
    """Define the Sigmoid module."""

    def __init__(self):
        """Initialize metadata."""
        super().__init__()
        self.name = "Sigmoid"
        self.type = "Activation"
        self.output = None

    def forward(self, data):
        """Compute a forward pass."""
        self.output = 1/(1 + (-data).exp())
        return self.output

    def backward(self, grad):
        """Compute a backward pass."""
        return grad*(self.output-self.output.pow(2))


class Sequential(Module):
    """Wrapper to combine multiple layers and activation functions easily."""

    def __init__(self, name, *args):
        """Initialize metadata."""
        super().__init__()
        # convert tuple of layers ot list of layers
        self.layers = list(args)
        self.name = name
        self.grad = None

    def __call__(self, data):
        """Allow for calling the module directly."""
        return self.forward(data)

    def forward(self, data):
        """Compute a forward pass."""
        self.data = data
        output = data
        for layer in self.layers:
            output = layer.forward(output)
#             print("After {} forward, output is {}.format(layer, output))
        return output

    def backward(self, upstream_derivative):
        """Compute a backward pass."""
        deriv = upstream_derivative
        for layer in reversed(self.layers):
            deriv = layer.backward(deriv)
        self.grad = deriv
        return self.grad

    @property
    def param(self):
        """Get parametes of all layers."""
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.param)
        return parameters

    def zero_grad(self):
        """Reset gradient for all layers."""
        for _, grad in self.param:
            grad.zero_()

    def reset(self):
        """Call the reset on all layers."""
        for layer in self.layers:
            layer.reset()

    def __repr__(self):
        """Print a representation of the MLP."""
        return ("Sequential MLP with {} layers: {}"
                .format(len(self.layers), str([str(l) for l in self.layers])))
