import warnings
from typing import Optional
import torch
import numpy as np
from sklearn.utils.random import sample_without_replacement

class NotFittedError(ValueError, AttributeError):
    """Raise Exception if estimator is used before fitting."""

def johnson_lindenstrauss_min_dim(n_samples: int, eps: float=0.1):
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int64)

class BaseRandomProjection:
    def __init__(self, n_components="auto", eps: float=0.1, random_state: Optional[int]=None) -> None:
        self.n_components = n_components
        self.n_components_: int
        self.random_matrix: torch.Tensor
        self.eps = eps
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)

    def _make_random_matrix(self, n_components: int, n_features: int):
        raise NotImplementedError

    def fit(self, embedding: torch.Tensor) -> "BaseRandomProjection":
        n_samples, n_features = embedding.shape
        if self.n_components == "auto":
            self.n_components_ = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)
            # if self.n_components_ <= 0 or self.n_components_ > n_features:
            if self.n_components_ <= 0:
                raise ValueError(f"Invalid target dimension {self.n_components_} for eps={self.eps} and n_samples={n_samples}.")
            elif self.n_components_ > n_features: # 允许升维
                self.n_components_ = n_features
        else:
            self._validate_n_components(n_features)
        self.random_matrix = self._make_random_matrix(self.n_components_, n_features).to(embedding.device)
        return self

    def _validate_n_components(self, n_features):
        if self.n_components <= 0:
            raise ValueError(f"n_components must be greater than 0, got {self.n_components}")
        elif self.n_components > n_features:
            warnings.warn("n_components is greater than the number of features. The dimensionality will not be reduced.")

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.random_matrix is None:
            raise NotFittedError("`fit()` has not been called.")
        projected_embedding = embedding @ self.random_matrix.T
        return projected_embedding


class SparseRandomProjection(BaseRandomProjection):
    def _make_random_matrix(self, n_components: int, n_features: int):
        density = 1 / np.sqrt(n_features)
        components = torch.zeros((n_components, n_features), dtype=torch.float64)
        if density == 1:
            components.normal_(0, 1 / np.sqrt(n_components))
        else:
            for i in range(n_components):
                nnz_idx = np.random.binomial(n_features, density)
                c_idx = sample_without_replacement(n_population=n_features, n_samples=nnz_idx, random_state=self.random_state)
                data = (np.random.rand(nnz_idx) < 0.5) * 2 - 1
                components[i, torch.tensor(c_idx, dtype=torch.long)] = torch.tensor(data, dtype=torch.float64)
            components *= np.sqrt(1 / density) / np.sqrt(n_components)
        return components.float()


class GaussianRandomProjection(BaseRandomProjection):
    def _make_random_matrix(self, n_components: int, n_features: int):
        components = torch.empty((n_components, n_features))
        torch.nn.init.normal_(components)
        return components


class SemiOrthogonalRandomProjection(BaseRandomProjection):
    def _make_random_matrix(self, n_components: int, n_features: int):
        components = torch.empty((n_components, n_features))
        torch.nn.init.orthogonal_(components)
        return components

if __name__ == "__main__":
    # Generate random data
    n_samples, n_features = 10000, 384
    data = torch.randn(n_samples, n_features)

    # Sparse Random Projection
    print("Testing SparseRandomProjection...")
    srp = SparseRandomProjection(random_state=42, eps=0.9)
    srp.fit(data)
    projected_data_srp = srp.transform(data)
    print(f"Sparse Random Projection Shape: {projected_data_srp.shape}")

    # Gaussian Random Projection
    print("Testing GaussianRandomProjection...")
    grp = GaussianRandomProjection(random_state=42, eps=0.9)
    grp.fit(data)
    projected_data_grp = grp.transform(data)
    print(f"Gaussian Random Projection Shape: {projected_data_grp.shape}")

    # Semi Orthogonal Random Projection
    print("Testing SemiOrthogonalRandomProjection...")
    sorp = SemiOrthogonalRandomProjection(random_state=42, eps=0.9)
    sorp.fit(data)
    projected_data_sorp = sorp.transform(data)
    print(f"Semi Orthogonal Random Projection Shape: {projected_data_sorp.shape}")

    print("All tests completed.")