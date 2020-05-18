"""Define useful functions."""
import math

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from optim import SGD


def one_hot_embedding(labels):
    """Do one-hot encoding of labels."""
    return torch.eye(2)[labels]


def generate_disc_set(size):
    """Generate `nb` data points with their labels."""
    data = Tensor(size, 2).uniform_(0, 1)
    # gosh that is dirty, but works and works fast
    labels = (data.sub(0.5)
              .pow(2)
              .sum(1)
              .sub(1/(2*math.pi))
              .sign()
              .sub(1)
              .abs()
              .div(2)
              .long()
              )
    return data, labels


def plot_preds(data, labels, axe=None, size=10):
    """Scatter the points in an ax with a color."""
    if axe is None:
        fig = plt.figure(figsize=(5, 5))
        axe = fig.subplots(1, 1)
    colors = ["blue" if lab == 0 else "red" for lab in labels]
    axe.scatter(data[:, 0], data[:, 1], c=colors, s=size)


def compute_nb_errors(model, data, targets):
    """Count the total of misclassified points."""
    nb_errors = 0
    output = model(data)
    predictions = torch.max(output.data, 1).indices

    for pred, targ in zip(predictions, targets):
        assert pred in [0, 1] and targ in [0, 1]
        if pred != targ:
            nb_errors = nb_errors + 1
    return nb_errors


def train_model(model, train_data, train_target, criterion,
                minibatch_size=50, nb_epochs=200, learning_rate=1e-3*0.5):
    """Define a wrapper to train a model easily."""
    optimizer = SGD(model.param, learning_rate=learning_rate)
    if train_target.shape[-1] != 2:
        # if not done yet, one-hot encore
        train_target = one_hot_embedding(train_target)
    loss_list = []
    for epoch in range(nb_epochs):
        size = train_data.shape[0]
        for batch in range(0, size, minibatch_size):
            # create subset of data/labels
            batch_input = train_data.narrow(0, batch, minibatch_size)
            batch_targets = train_target.narrow(0, batch, minibatch_size)

            predictions = model.forward(batch_input)
            loss = criterion(predictions, batch_targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.loss.item())
        if epoch % (nb_epochs//10) == 0 or epoch == nb_epochs-1:
            print("Epoch {}, loss : {}".format(epoch+1, loss_list[-1]))
    return loss_list, model
