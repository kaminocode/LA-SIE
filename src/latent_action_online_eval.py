from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.latent_action import apply_operator


def _scalar_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _effective_count(reference: torch.Tensor, max_samples: int) -> int:
    if reference.shape[0] == 0:
        return 0
    return min(reference.shape[0], max(max_samples, 0))


def _pairwise_action_distance(source_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    source_z = source_z.float()
    target_z = target_z.float()
    if source_z.shape[1] == 4 and target_z.shape[1] == 4:
        source_z = F.normalize(source_z, dim=1)
        target_z = F.normalize(target_z, dim=1)
        return 1.0 - (source_z @ target_z.T).abs()
    return torch.cdist(source_z, target_z)


def _best_cross_image_matches(
    donor_z: torch.Tensor,
    recipient_z: torch.Tensor,
    donor_labels: torch.Tensor,
    recipient_labels: torch.Tensor,
    *,
    same_class: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    distances = _pairwise_action_distance(donor_z, recipient_z)
    label_match = donor_labels.unsqueeze(1) == recipient_labels.unsqueeze(0)
    class_mask = label_match if same_class else ~label_match

    if same_class:
        donor_count = donor_z.shape[0]
        class_mask[torch.arange(donor_count, device=donor_z.device), torch.arange(donor_count, device=donor_z.device)] = False

    distances = distances.masked_fill(~class_mask, torch.inf)
    matched_indices = distances.argmin(dim=1)
    matched_distances = distances[
        torch.arange(matched_indices.shape[0], device=matched_indices.device),
        matched_indices,
    ]
    return matched_indices, torch.isfinite(matched_distances)


def _matched_transfer_error(
    donor_operators: torch.Tensor,
    recipient_sources: torch.Tensor,
    recipient_targets: torch.Tensor,
    matched_indices: torch.Tensor,
) -> torch.Tensor:
    if matched_indices.numel() == 0:
        return _scalar_zero(donor_operators)

    valid_indices = matched_indices.long()
    recipient_sources = recipient_sources.index_select(0, valid_indices)
    recipient_targets = recipient_targets.index_select(0, valid_indices)
    transported = apply_operator(donor_operators, recipient_sources)
    return F.mse_loss(transported, recipient_targets.float())


def _transfer_metric(
    donor_operators: torch.Tensor,
    donor_z: torch.Tensor,
    donor_labels: torch.Tensor,
    recipient_sources: torch.Tensor,
    recipient_targets: torch.Tensor,
    recipient_z: torch.Tensor,
    recipient_labels: torch.Tensor,
    *,
    same_class: bool,
) -> torch.Tensor:
    if donor_operators.shape[0] == 0:
        return _scalar_zero(recipient_sources)

    matched_indices, valid_mask = _best_cross_image_matches(
        donor_z,
        recipient_z,
        donor_labels,
        recipient_labels,
        same_class=same_class,
    )
    if not torch.any(valid_mask):
        return _scalar_zero(recipient_sources)

    return _matched_transfer_error(
        donor_operators[valid_mask],
        recipient_sources,
        recipient_targets,
        matched_indices[valid_mask],
    )


@torch.no_grad()
def cross_image_transfer_stats(
    args,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    prediction,
    labels: torch.Tensor,
    z: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if not getattr(args, "latent_online_eval", False):
        return {}

    labels = labels.reshape(-1)
    donor_count = _effective_count(
        source_features,
        getattr(args, "latent_online_eval_samples", 16),
    )
    donor_operators = prediction.operator[:donor_count]
    donor_labels = labels[:donor_count]
    donor_z = z[:donor_count]

    return {
        "LatentEval/transfer_same_class_mse": _transfer_metric(
            donor_operators,
            donor_z,
            donor_labels,
            source_features,
            target_features,
            z,
            labels,
            same_class=True,
        ),
        "LatentEval/transfer_diff_class_mse": _transfer_metric(
            donor_operators,
            donor_z,
            donor_labels,
            source_features,
            target_features,
            z,
            labels,
            same_class=False,
        ),
    }
