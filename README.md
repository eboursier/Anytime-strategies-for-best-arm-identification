# Presentation of the work

This work was done for a review project of the Reinforcement Learning course of the MVA Master (ENS Paris-Saclay). The purpose of the code is globally to implement the main algorithms for anytime strategies for best arm identification and compare them on some basic examples.

The project and the code was done in collaboration with Zineb Belkacemi.

# Description of the code

errors.py implements all the different scores/errors that could be used to quantify the accuracy of a list of recommendations
arms.py implements all the arms and MAB classes
all the other .py implement different used sampling strategies

First create a scores and a figures folder in your root folder. Then run the Experiments notebook.
It will automatically save your obtained scores and figures while running your code, so you won't have to train several times, since it can take some time. Also, if you interrupt your simulation, it will save the current state of the simulation.

requires:
- python 3
- numpy
- matplotlib
- time
- tqdm
