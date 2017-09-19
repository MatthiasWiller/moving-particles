"""
# ---------------------------------------------------------------------------
# Seed Selector Class for Moving Particles
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
# ---------------------------------------------------------------------------
# Input:
# * strategy          : seed selection strategy
#                     -> 0: always select active path as seed
#                     -> 1: no blocking, all samples are available as seeds
#                     -> 2: block only active path
#                     -> 3: block active path and sons
#                     -> 4: block active path and father
#                     -> 5: block active path, father and sons
# * number_of_samples : number of samples for Moving Particles algorithm
# ---------------------------------------------------------------------------
# Output:
# * seed_id   : ID of the seed that will be selected for this MCMC step
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

class SeedSelector:
    def __init__(self, strategy, number_of_samples):
        self.strategy = strategy
        self.number_of_samples = int(number_of_samples)
        # Seed matrix --> 0: blocked, 1: available
        self.seed_matrix = np.ones((self.number_of_samples, self.number_of_samples))

        if self.strategy in [2, 3, 4, 5]:
            eye = np.identity(self.number_of_samples)
            self.seed_matrix = self.seed_matrix - eye


    def get_seed_id(self, active_path_id):
        # if all samples are blocked, release them
        # and block just sample of active path
        if np.sum(self.seed_matrix[active_path_id, :]) == 0:
            self.seed_matrix = np.identity((self.number_of_samples, self.number_of_samples))

        seed_list = []
        for i in range(0, self.number_of_samples):
            if self.seed_matrix[active_path_id, i] == 1:
                seed_list.append(i)

        seed_id = np.random.choice(seed_list)

        if self.strategy == 0:
            # always select active path
            seed_id = active_path_id

        self.update_seed_matrix(active_path_id, seed_id)

        return seed_id


    def update_seed_matrix(self, active_path_id, seed_id):
        # unblock active path, because it was updated
        self.seed_matrix[active_path_id, :] = np.ones((1, self.number_of_samples))
        self.seed_matrix[:, active_path_id] = np.ones((self.number_of_samples))

        # strategy == 0: don't have to do anything here
        # strategy == 1: don't have to do anything here

        if self.strategy == 2:
            # block only active path
            self.seed_matrix[active_path_id, active_path_id] = 0

        elif self.strategy == 3:
            # block active path and son
            self.seed_matrix[active_path_id, active_path_id] = 0
            self.seed_matrix[active_path_id, seed_id] = 0

        elif self.strategy == 4:
            # block active path and father
            self.seed_matrix[active_path_id, active_path_id] = 0
            self.seed_matrix[seed_id, active_path_id] = 0

        elif self.strategy == 5:
            # block active path, father and sons
            self.seed_matrix[active_path_id, active_path_id] = 0
            self.seed_matrix[seed_id, active_path_id] = 0
            self.seed_matrix[active_path_id, seed_id] = 0
