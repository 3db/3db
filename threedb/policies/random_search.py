"""
threedb.policies.random_search
==============================

A search policy over controls that randomly samples from the set of 
permissible controls.
"""

import numpy as np
from itertools import product

class RandomSearchPolicy:
    def __init__(self, continuous_dim, discrete_sizes, samples, seed=None):
        """
            Pick a total number of samples randomly
        """
        self.continuous_dim = continuous_dim
        self.discrete_sizes = discrete_sizes
        self.samples = samples
        self.seed = seed

    def hint_scheduler(self):
        return 1, self.samples 

    def run(self, render_and_send):
        rng = np.random.default_rng(self.seed)
        result = []
        for _ in range(self.samples):
            continuous_instance = rng.random(self.continuous_dim)
            discrete_instance = [rng.integers(low=0, high=n) for n in self.discrete_sizes]
            result.append((continuous_instance, discrete_instance))

        render_and_send(result)

Policy = RandomSearchPolicy