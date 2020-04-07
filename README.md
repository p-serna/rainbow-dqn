# Rainbow Deep Q-Network

## About

This is a repository for developing a Rainbow  DQN. The starting seed is a miniproject from Udacity's Nanodegree in Reinforcement Learning. Implemented:

- Deep Q-Network with memory replay for learning (DQN)

- Double DQN (*for some reason it still does not learn*)

- Duel (Double) DQN (when not using DDQN it learns well, although a bit slower than the DQN)

- Importance sampling

The architecture of the network can be specified when instantiating the agent itself with a list of number of units (for fully-connected layers).

To see the application in another game: [Picking up bananas](https://github.com/pabloserna/Navigation-banana)

## Get started

Python files Model and dqn_agent contain the definition of the model and the agents. The jupyter notebook shows how to use and train them 

## Requirements

- Numpy

- Matplotlib

- OpenAI Gym (LunarLander-v2)

- Pytorch

- pyvirtualdisplay

- box2d

- xvfb
