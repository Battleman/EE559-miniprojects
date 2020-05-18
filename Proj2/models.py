"""This file declares the models to be used for testing."""

from modules import Sequential, Linear, ReLU, Tanh, Sigmoid

MODEL1 = Sequential("ReLu",
                    Linear(2, 25), ReLU(),
                    Linear(25, 25), ReLU(),
                    Linear(25, 25), ReLU(),
                    Linear(25, 2), Sigmoid()
                    )

MODEL2 = Sequential("Tanh",
                    Linear(2, 25), Tanh(),
                    Linear(25, 25), Tanh(),
                    Linear(25, 25), Tanh(),
                    Linear(25, 2), Sigmoid()
                    )

MODEL3 = Sequential("ReLu + He",
                    Linear(2, 25, "He"), ReLU(),
                    Linear(25, 25, "He"), ReLU(),
                    Linear(25, 25, "He"), ReLU(),
                    Linear(25, 2, "He"), Sigmoid()
                    )

MODEL4 = Sequential("Tanh + Xavier",
                    Linear(2, 25, "Xavier"), Tanh(),
                    Linear(25, 25, "Xavier"), Tanh(),
                    Linear(25, 25, "Xavier"), Tanh(),
                    Linear(25, 2, "Xavier"), Sigmoid()
                    )

# Best model is actually almost model 2
MODEL_BEST = Sequential("Best",
                        Linear(2, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 25), Tanh(),
                        Linear(25, 2, "He"), Sigmoid()
                        )
