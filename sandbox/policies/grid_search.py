import numpy as np
from itertools import product


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
        continuous_values = np.linspace(0, 1, self.samples_per_dim)

        discrete_spaces = []

        for n in self.discrete_sizes:
            discrete_spaces.append(np.arange(n))

        result = []
        for continuous_instance in product(*([continuous_values] * self.continuous_dim)):
            for discrete_instance in product(*discrete_spaces):
                result.append((continuous_instance, discrete_instance))

        images, logits = render(result)
        print(images.shape)
        print(logits.shape)


Policy = GridSearchPolicy
