"""Define the models to be trained."""
from torch import nn, optim
from torch.nn import functional as F


class BaseNetClass(nn.Module):
    """The mother of all Networks."""

    def __init__(self, auxiliary=False):
        """Set common metadata for all Networks models."""
        super().__init__()
        self.auxiliary = auxiliary

    def train_model(self, train_data, train_target, train_classes=None,
                    nb_epochs=25, lambda_l2=0.1, batch_size=100, lr=1e-3*0.5):
        """Defina a wrapper to train, for all models."""
        if self.auxiliary:
            if train_classes is None:
                print("Auxiliary loss requires train classes.\
                    Please set train_classes")
                raise ValueError(
                    "Requested auxiliary loss without training classes")
            self.auxiliary = True
        loss_list = []
        for _ in range(nb_epochs):
            size = train_data.size(0)
            for batch in range(0, size, batch_size):
                loss = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self.parameters(), lr=lr)
                batch_input = train_data.narrow(0,
                                                batch,
                                                batch_size)
                batch_classes = train_classes.narrow(0,
                                                     batch,
                                                     batch_size)
                batch_targets = train_target.narrow(0,
                                                    int(batch/2),
                                                    int(batch_size/2))

                # forward pass, get classifications
                classif2, classif10 = self(batch_input)
                output = loss(classif2, batch_targets)  # compute error
                if self.auxiliary:
                    # only use classif10 is we want auxiliary loss
                    output += loss(classif10, batch_classes)

                if lambda_l2 is not None:
                    # Add L2-regularization
                    for param in self.parameters():
                        output += lambda_l2 * param.pow(2).sum()

                # reset gradients, backward pass, optimizer step
                self.zero_grad()
                output.backward()
                optimizer.step()

                # log the loss
                loss_list.append(output.data.item())
        return loss_list


class Baseline(BaseNetClass):
    """Baseline neural network, without weight sharing."""

    def __init__(self, auxiliary=False):
        """Define the main components of the Neural Net."""
        super().__init__(auxiliary)
        self.fc1 = nn.Linear(196*(1+(not auxiliary)), 200)
        self.bn1 = nn.BatchNorm1d(200)

        ####
        # dropout half 200->100
        ####

        self.fc2 = nn.Linear(100, 256)
        self.bn2 = nn.BatchNorm1d(256)

        ####
        # dropout half 256->128
        ####

        # predict
        self.fc_final_10 = nn.Linear(128, 10)
        self.fc_final_2 = nn.Linear(20, 2)

    def forward(self, x):
        """Define the forward pass."""
        # pylint: disable=C0103
        N = x.shape[0]

        y = self.fc1(x.view(N, -1))
        y = self.bn1(y)
        y = F.relu(y)
        y = F.max_pool1d(y.view(N, 1, -1),
                         kernel_size=2,
                         stride=2)

        y = F.dropout(y, p=0.5)

        y = self.fc2(y.view(N, -1))
        y = self.bn2(y)
        y = F.relu(y)
        y = F.max_pool1d(y.view(N, 1, -1), kernel_size=2)

        y = F.dropout(y, p=0.5)

        classif10 = self.fc_final_10(y).view(N, -1)
        # important to use shape[0]//2: because 2 channels
        classif2 = F.relu(self.fc_final_2(
            classif10.view(classif10.shape[0]//2, -1)))
        # pylint: enable C0103
        return classif2, classif10


class CNN(BaseNetClass):
    """Define the network class with convolutions."""

    def __init__(self, auxiliary=False):
        """Define a basic convolutional Neural Network."""
        super().__init__(auxiliary)
        # layer 1
        self.conv1 = nn.Conv2d(1+(not auxiliary), 16, kernel_size=5, padding=3)
        self.bn1 = nn.BatchNorm2d(16)

        # layer 2
        self.conv2 = nn.Conv2d(16, 20, kernel_size=5, padding=3)
        self.bn2 = nn.BatchNorm2d(20)

        # layer 3
        self.fc1 = nn.Linear(500, 300)

        # classif10
        self.fc2 = nn.Linear(300, 10)

        # classif2
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        """Define forward pass.

        Basically: conv->batch_norm->relu->max_pool2d
        """
        # layer 1
        # pylint: disable=C0103
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = F.max_pool2d(y, kernel_size=2)

        # layer 2
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = F.max_pool2d(y, kernel_size=2)

        # layer 3
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = F.relu(y)

        # generate class (0-9)
        classif10 = self.fc2(y)

        # relu the classification, and down to 2
        y = self.fc3(classif10)
        y = F.relu(y)
        classif2 = y.view(int(classif10.shape[0]/2), -1)
        # pylint: enable=E1101
        return classif2, classif10
