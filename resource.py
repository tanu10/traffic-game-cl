import numpy as np


class Resource:

    def __init__(self, rid, mu, sigma):
        self.id = rid
        self.mu = mu
        self.sigma = sigma

    def get_cost(self, num_agents):
        return num_agents * np.exp(- pow((num_agents-self.mu)/self.sigma, 2))
