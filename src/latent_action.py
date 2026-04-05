import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _default_hidden_dim(feature_dim, output_dim):
    return max(feature_dim, output_dim * 2)


def _skew_symmetric(matrix):
    return 0.5 * (matrix - matrix.transpose(-1, -2))


def _identity_batch(reference, dim):
    eye = torch.eye(dim, device=reference.device, dtype=reference.dtype)
    return eye.unsqueeze(0).expand(reference.shape[0], -1, -1)


@dataclass
class OperatorPrediction:
    operator: torch.Tensor
    code: Optional[torch.Tensor] = None
    coefficients: Optional[torch.Tensor] = None
    raw_matrix: Optional[torch.Tensor] = None


class PairMLP(nn.Module):
    def __init__(self, feature_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or _default_hidden_dim(feature_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h_i, h_j):
        return self.net(torch.cat([h_i, h_j], dim=1))


class DirectFullMatrixOperator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = PairMLP(feature_dim, feature_dim * feature_dim, hidden_dim=hidden_dim)

    def forward(self, h_i, h_j):
        matrix = self.head(h_i, h_j).view(-1, self.feature_dim, self.feature_dim)
        matrix = matrix.float() / math.sqrt(float(self.feature_dim))
        return OperatorPrediction(operator=matrix, raw_matrix=matrix)


class DirectSkewExpOperator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = PairMLP(feature_dim, feature_dim * feature_dim, hidden_dim=hidden_dim)

    def forward(self, h_i, h_j):
        raw_matrix = self.head(h_i, h_j).view(-1, self.feature_dim, self.feature_dim)
        raw_matrix = raw_matrix.float() / math.sqrt(float(self.feature_dim))
        skew_matrix = _skew_symmetric(raw_matrix)
        operator = torch.matrix_exp(skew_matrix)
        return OperatorPrediction(operator=operator, raw_matrix=skew_matrix)


class LatentCodeToFullMatrixOperator(nn.Module):
    def __init__(self, feature_dim, code_dim, hidden_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        hidden_dim = hidden_dim or _default_hidden_dim(feature_dim, code_dim)
        self.code_encoder = PairMLP(feature_dim, code_dim, hidden_dim=hidden_dim)
        self.matrix_head = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * feature_dim),
        )

    def forward(self, h_i, h_j):
        code = self.code_encoder(h_i, h_j)
        matrix = self.matrix_head(code).view(-1, self.feature_dim, self.feature_dim)
        matrix = matrix.float() / math.sqrt(float(self.feature_dim))
        return OperatorPrediction(operator=matrix, code=code.float(), raw_matrix=matrix)


class SharedGeneratorOperator(nn.Module):
    def __init__(
        self,
        feature_dim,
        code_dim,
        num_generators,
        *,
        learnable_generators,
        hidden_dim=None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.code_dim = code_dim
        self.num_generators = num_generators
        hidden_dim = hidden_dim or _default_hidden_dim(feature_dim, code_dim)
        self.code_encoder = PairMLP(feature_dim, code_dim, hidden_dim=hidden_dim)
        if code_dim == num_generators:
            self.coefficient_head = nn.Identity()
        else:
            self.coefficient_head = nn.Linear(code_dim, num_generators, bias=False)

        raw_generators = torch.randn(num_generators, feature_dim, feature_dim)
        raw_generators = _skew_symmetric(raw_generators) / math.sqrt(float(feature_dim))
        if learnable_generators:
            self.generator_params = nn.Parameter(raw_generators)
        else:
            self.register_buffer("generator_params", raw_generators)

    def generator_matrices(self):
        return _skew_symmetric(self.generator_params.float())

    def forward(self, h_i, h_j):
        code = self.code_encoder(h_i, h_j).float()
        coefficients = self.coefficient_head(code)
        raw_matrix = torch.einsum("bk,kij->bij", coefficients, self.generator_matrices())
        operator = torch.matrix_exp(raw_matrix)
        return OperatorPrediction(
            operator=operator,
            code=code,
            coefficients=coefficients.float(),
            raw_matrix=raw_matrix,
        )


def apply_operator(operator, features):
    return torch.bmm(operator.float(), features.float().unsqueeze(-1)).squeeze(-1)


def operator_norm(operator):
    return operator.float().pow(2).sum(dim=(-2, -1)).sqrt().mean()


def vector_norm(vector):
    return vector.float().pow(2).sum(dim=-1).sqrt().mean()


def identity_error(operator):
    return F.mse_loss(operator.float(), _identity_batch(operator, operator.shape[-1]))


def inverse_error(forward_operator, backward_operator):
    identity = _identity_batch(forward_operator, forward_operator.shape[-1])
    return F.mse_loss(torch.bmm(backward_operator.float(), forward_operator.float()), identity)


def composition_error(direct_operator, composed_left, composed_right):
    composed = torch.bmm(composed_left.float(), composed_right.float())
    return F.mse_loss(direct_operator.float(), composed)
