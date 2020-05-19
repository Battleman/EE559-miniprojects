#!/usr/bin/env python3
"""Create a testing bench callable without arguments."""
from models import Baseline, CNN
from pipelines import pipeline

ROUNDS = 20
EPOCHS = 25
MBS = 100
LR = 1e-3
pipeline(Baseline, "Baseline", rounds=ROUNDS, epochs=EPOCHS,
         mini_batch_size=MBS, lr=LR)

pipeline(Baseline, "Baseline with auxiliary", rounds=ROUNDS, epochs=EPOCHS,
         mini_batch_size=MBS, lr=LR, auxiliary=True)

LR = 5e-4
pipeline(CNN, "CNN", rounds=ROUNDS, epochs=EPOCHS,
         mini_batch_size=MBS, lr=LR)

pipeline(CNN, "CNN with auxiliary", rounds=ROUNDS, epochs=EPOCHS,
         mini_batch_size=MBS, lr=LR, auxiliary=True)
