#!/usr/bin/env python3
"""Define the standard test."""
import torch

from criterion import LossMSE
from models import MODEL_BEST
from util import compute_nb_errors, train_model, generate_disc_set

torch.set_grad_enabled(False)  # VERY IMPORTANT

######
# Create training and testing data
######
TRAINING_DATA, TRAINING_LABELS = generate_disc_set(1000)
TESTING_DATA, TESTING_LABELS = generate_disc_set(1000)


#########
# Global training parameters
#########
NB_EPOCHS = 300
LR = 0.05
MINIBATCH_SIZE = 100
MODEL = MODEL_BEST
CRITERION = LossMSE(MODEL)


# Computing
print("##### Training model#####")
_, MODEL_TRAINED = train_model(MODEL, TRAINING_DATA, TRAINING_LABELS,
                               criterion=CRITERION, learning_rate=LR,
                               nb_epochs=NB_EPOCHS,
                               minibatch_size=MINIBATCH_SIZE)

TRAINING_ERROR = compute_nb_errors(MODEL_TRAINED,
                                   TRAINING_DATA,
                                   TRAINING_LABELS)
TESTING_ERROR = compute_nb_errors(MODEL_TRAINED,
                                  TESTING_DATA,
                                  TESTING_LABELS)
print("***** Errors *****")
print("Training: {:.4f}%".format(100*TRAINING_ERROR/len(TRAINING_DATA)))
print("Testing: {:.4f}%".format(100*TESTING_ERROR/len(TESTING_DATA)))
print("******************\n\n")
