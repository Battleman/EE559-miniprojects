"""Define classes relative to optimization."""


class SGD():
    """Stochastic Gradient Descent Optimizer class."""

    def __init__(self, param, learning_rate=0.01):
        """Initialize SGD."""
        self.params = param
        self.learning_rate = learning_rate

    def step(self):
        """Perform a single optimization step."""
        for parameter, p_grad in self.param:
            # update parameter
            parameter.add_(-self.learning_rate * p_grad)

    @property
    def param(self):
        """Return all parameters."""
        return self.params
