import numpy as np
from bayessb import convergence

class MCMCSet(object):
    """Class for storage and management of multiple MCMC objects representing
    repeated runs of the MCMC."""

    def __init__(self, name):
        """Create the MCMCSet object and assign a name.

        Assigns a name to the set and initializes an empty list
        of MCMC objects.

        Parameters
        ----------
        name : string
            The string describing the model/name/data/mcmc parameters that
            is used to identify the set of chains.
        """

        self.name = name
        self.chains = []
        self.pooled_positions = None

    def add_chain(self, chain):
        """Add an MCMC chain to the set."""
        self.chains.append(chain)

    def prune_all_chains(self, burn, thin=1):
        """Iterates over all the chains and prunes each one with the
        specified arguments.
        """

        for chain in self.chains:
            chain.prune(burn, thin)

    def all_pruned(self):
        """Indicates whether all chains have been pruned already.
        """

        if not self.chains:
            raise Exception("There are no chains in the MCMCSet.")

        for chain in self.chains:
            if not chain.pruned:
                return False
        return True

    def pool_chains(self):
        """Pool the chains into a single set of pooled positions stored along
        with the MCMCSet.
        """

        if not self.chains:
            raise Exception("There are no chains in the MCMCSet.")

        # First, count the total number of steps after pruning and make sure
        # all chains have been pruned.
        total_positions = 0
        for chain in self.chains:
            if not chain.pruned:
                raise Exception("The chains have not yet been pruned.")
            else:
                total_positions += len(chain.positions)

        # Allocate enough space for the pooled positions
        self.pooled_positions = np.zeros((total_positions,
                                          self.chains[0].num_estimate))

        # Iterate again, filling in the pooled positions
        start_index = 0
        for chain in self.chains:
            last_index = start_index + len(chain.positions)
            self.pooled_positions[start_index:last_index,:] = chain.positions
            start_index = last_index

    def get_sample_position(self):
        """Returns a position sampled at random from the pooled chains.

        Requires that the chains have already been pooled. 
        """
        if not self.chains:
            raise Exception("There are no chains in the MCMCSet.")

        if self.pooled_positions is None:
            raise Exception("Cannot get a sample position until the chains " \
                            "have been pooled.")

        rand_index = np.random.randint(len(self.pooled_positions))
        return self.pooled_positions[rand_index]

    def get_sample_simulation(self, observables=True):
        """Uses the model in the first chain in the set to run a simulation for
        a randomly sampled position from the pooled chains.
        """

        position = self.get_sample_position()
        return self.chains[0].simulate(position=position, observables=True)

    def initialize_and_pool(self, chains, burn, thin=1):
        """Adds the chains to the MCMCSet and prunes and pools them."""
        for chain in chains:
            self.add_chain(chain)

        self.prune_all_chains(burn, thin)
        self.pool_chains()

