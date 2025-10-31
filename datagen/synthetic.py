from typing import override

import hydra
import numpy as np
import scipy.sparse as sp

from neural_cg.datagen_helper import DatagenBase


def generate_spd_sparse_matrix(n, sparsity=0.01, condition_amplifier=1e-6, random_state=None):
    rng = np.random.default_rng(random_state)

    # Step 1: Generate a sparse random matrix M
    M = sp.random(n, n, density=sparsity, format='csr', random_state=rng)
    M.data = (M.data - 0.5) * 2  # Scale to [-1, 1]

    # Step 2: Make the matrix more ill-conditioned
    # Apply column scaling to introduce anisotropy
    scaling = np.linspace(1, condition_amplifier, n)
    D = sp.diags(scaling)
    M = D @ M  # Right-multiply to skew column scales

    # Step 3: Form A = MᵀM + αI to ensure SPD
    A = M.T @ M
    A += sp.eye(n) * condition_amplifier  # Small regularizer to ensure strict SPD

    return A

class SyntheticDatagen(DatagenBase):
    def __init__(self, config):
        super().__init__(**config.basic)

        self.random_state = config.random_state
        self.sparsity = config.sparsity
        self.alpha = config.algebra.epsilon
        self.lower_size = config.algebra.low
        self.upper_size = config.algebra.high
        self.rng = np.random.RandomState(self.random_state)

    @override
    def step(self):
        msize = np.random.randint(self.lower_size, 1 + self.upper_size)
        A = generate_spd_sparse_matrix(
            n=msize,
            sparsity=self.sparsity,
            condition_amplifier=self.alpha,
            random_state=self.rng,
        )
        return A, None, None, None


@hydra.main(config_path="config", config_name="synthetic", version_base="1.3")
def main(cfg):
    print(cfg)
    datagen = SyntheticDatagen(cfg)
    datagen.generate()

if __name__ == "__main__":
    main()