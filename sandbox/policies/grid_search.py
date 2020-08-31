import numpy as np


class GridSearchPolicy:

    def __init__(self, continuous_dim, discrete_sizes, samples_per_dim):
        self.continuous_dim = continuous_dim
        self.discrete_sizes = discrete_sizes
        self.samples_per_dim = samples_per_dim

    def hint_scheduler(self):
        total_queries = self.samples_per_dim ** self.continuous_dim
        total_queries *= np.prod(self.discrete_sizes)
        return 1, int(total_queries)

    def run(self, render):
        pass


Policy = GridSearchPolicy
