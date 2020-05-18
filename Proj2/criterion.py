"""Declare all classes relative to optimization.

Currently MSE and SGD. Possible to add others, such as CrossEntropy."""

import torch


class Optimizer():
    """Generic optimizer class"""

    def __init__(self, model):
        """Initialize the optimizer."""
        self.name = "Unknown"
        self.type = "Optimizer"
        self.model = model

        self.output = None
        self.target = None
        self.softmax = None
        self.loss = None

    def forward(self, output, target):
        """Do a forward pass."""
        raise NotImplementedError

    def backward(self):
        """Do a backward pass.

        Raises:
            NotImplementedError: Must implement this
        """
        raise NotImplementedError

    def __call__(self, output, target):
        """Allow to call the optimizer directly."""
        self.forward(output, target)
        return self

    def __repr__(self):
        """Make the optimizer printable."""
        return "{} {}".format(self.type, self.name)


class LossMSE(Optimizer):
    """Define a class for computing the loss through MSE."""

    def __init__(self, model):
        """Initialize the optimizer."""
        super().__init__(model)
        self.name = "MSE"
        self.err = None

    def forward(self, output, target):
        """Compute a forward pass."""
        self.output, self.target = output, target
        self.err = self.output - self.target  # compute error ~ difference
        self.loss = self.err.pow(2).sum() / len(self.err)  # compute MSE
        return self

    def backward(self):
        """Compute a backward pass."""
        dldx = 2*self.err/len(self.err)
        return self.model.backward(dldx)


class CrossEntropy(Optimizer):
    """Define the CrossEntropy optimizer."""

    def __init__(self, model):
        """Initialize the optimizer."""
        super().__init__(model)
        self.name = "CrossEntropy"

    def forward(self, output, target):
        """Compute and return mean loss."""
        self.output, self.target = output, target
        self.softmax = output.softmax(1)
        self.loss = (-target * (self.softmax.log())).sum(1).mean()
        return self

    def backward(self):
        """Compute a backward pass."""
        size = self.output.size()[0]
        diff = self.target[:, 1] * self.softmax[:, 0]  # take first difference
        # substract second difference
        diff -= -self.target[:, 0] * self.softmax[:, 1]
        diff = diff.view(-1, 1)  # change to colum view
        grad = torch.cat([diff, -diff], 1) / size  # concatenate to 2 columns
        # use this as gradient for backward step
        return self.model.backward(grad)
