"""Define all methods relative to a nice development pipeline."""
from time import time

import torch
from torch.nn import functional as F

import dlc_practical_prologue as prologue
from stats import mean, stddev


def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    """Compute the number of errors a model make.

    Take the data input, pass it through the model for predictions, and compare
    to the actual targets.

    Returns:
        positive integer, the absolute count of misclassified elements.
    """
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        # Prediction
        output_single, _ = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(F.softmax(output_single.data, 0), 1)

        # Compare and count error
        for k in range(int(mini_batch_size/2)):
            if data_target.data[int(b/2) + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def pipeline(net_class, name, rounds=2, epochs=25, mini_batch_size=50,
             lr=1e-3*0.5, auxiliary=False, **model_kwargs):
    """
    Full data generations, training, testing multiple times.

    For a given situation (mini-batch size, network, number of samples) does:
        * Data Generation
        * Model training (logging of losses)
        * Testing on training data (logging accuracy)
        * Testing on testing data (logging accuracy)
        * Printing results

    Parameters:
        net_class {nn.Module}: The class to train.
        rounds {int; default=10}: how many time the pipeline must be ran.
        epochs {int; default=25}: Number of epochs for the training.
        mini_batch_size {int; default=50}: size of the mini-batch.
            Must be a divisor of N.

    Returns:
        model: the model trained
        train_losses: List of training losses for each epoch.
            Shape: (rounds, num_epochs)
        train_errors: Mean errors per round (on a scale [0-1]) on train data.
            Shape: (rounds)
        test_errors: Mean errors per round (on a scale [0-1]) on test data.
            Shape: (rounds)
        rounds_times: Total time (data gen, training, testint) for each round.
            Shape: (rounds)
    """
    N = 1000
    train_losses = []
    train_errors, test_errors = [], []
    rounds_times = []
    print("********************* Training model '{}'*********************"
          .format(name))
    for i in range(rounds):
        print('Starting iteration {}'.format(i+1))
        time_start = time()
        # Generate Data
        (train_data, train_target, train_classes,
         test_data, test_target, test_classes) = prologue.generate_pair_sets(N)

        if auxiliary:
            # unpack the images to [2N, 1, 14, 14]
            train_data = train_data.view(-1, 1, 14, 14)
            train_classes = train_classes.view(-1)
            test_data = test_data.view(-1, 1, 14, 14)
            test_classes = test_classes.view(-1)

        # Model training
        time_train = time()
        model = net_class(auxiliary=auxiliary, **model_kwargs)
        train_epoch_losses = model.train_model(train_data,
                                               train_target,
                                               train_classes,
                                               nb_epochs=epochs,
                                               batch_size=mini_batch_size,
                                               lr=lr)
        train_losses.append(train_epoch_losses)

        # Compute train error
        time_train_errors = time()
        nb_train_errors = compute_nb_errors(model, train_data,
                                            train_target, mini_batch_size)
        train_errors.append(nb_train_errors/train_data.shape[0])

        # Compute train error
        time_test_errors = time()
        nb_test_errors = compute_nb_errors(model, test_data,
                                           test_target, mini_batch_size)
        test_errors.append(nb_test_errors/test_data.shape[0])

        time_end = time()
        rounds_times.append(time_end-time_start)
        # Logging
        print('Train error: {:0.2f}%'.format(
            100 * (nb_train_errors / train_data.size(0))))
        print('Test error: {:0.2f}%'
              .format(100 * (nb_test_errors / test_data.size(0))))
        print("Times:\n\
            Data generation: {:.2f}s\tTraining: {:.2f}\n\
            Errors compute train: {:.2f}s\tErrors compute test: {:.2f}s\n\
            Full round: {:.2f}s"
              .format(time_train-time_start,
                      time_train_errors-time_train,
                      time_test_errors-time_train_errors,
                      time_end-time_test_errors,
                      time_end-time_start))
        print("\n############\n")
    digest(train_errors, test_errors, rounds_times)
    print("******************* Ending training of '{}'*******************\n\n"
          .format(name))
    return model, train_losses, train_errors, test_errors, rounds_times


def digest(train_errors, test_errors, rounds_times):
    """Create a digest of results.

    Will print mean and std for all three lists, in a nicely formatted way.
    Arguments:
        train_errors {list} -- List of nb errors on training data
        test_errors {list} -- List of nb errors on testing data
        rounds_times {list} -- List of times to complete rounds
    """
    print("#*#*#*#*#*#*#*")
    print("Some stats accross rounds:")
    print("Train error: {:.4f}% +- {:.4f}".format(100*mean(train_errors),
                                                  100*stddev(train_errors)))
    print("Test error: {:.4f}% +- {:.4f}".format(100*mean(test_errors),
                                                 100*stddev(test_errors)))
    print("Round times: {:.4f}s +- {:.4f}s".format(mean(rounds_times),
                                                   stddev(rounds_times)))
    print("#*#*#*#*#*#*#*")
