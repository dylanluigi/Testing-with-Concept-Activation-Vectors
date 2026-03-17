"""Metrics for TCAV evaluation.

Three families of objective metrics for assessing the fidelity of
Visual-TCAV concept attributions and concept maps:

1. Aggregation helpers — aggregate_tcav_runs
2. C-Insertion / C-Deletion (Appendix K, De Santis et al. TMLR)
3. Concept Map Localization (Appendix H, De Santis et al. TMLR)
4. Saliency Distribution Metrics (Miró-Nicolau et al., 2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

# =========
# Aggregation
# =========

def aggregate_tcav_runs(runs: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Average TCAV scores across multiple runs (nanmean per layer/concept)."""
    out: Dict[str, Dict[str, float]] = {}
    if not runs:
        return out
    for layer in runs[0].keys():
        all_concepts = runs[0][layer].keys()
        out[layer] = {}
        for c in all_concepts:
            vals = [r[layer][c] for r in runs if c in r[layer]]
            out[layer][c] = float(np.nanmean(vals)) if vals else float("nan")
    return out

# =========
# Data classes
# =========

@dataclass
class CIDResult:
    """Results from a single C-Insertion/C-Deletion run for one concept-class pair.

    Attributes:
        concept_name:   str — concept being evaluated.
        class_index:    int — target class index.
        t_grid:         np.ndarray, shape [S] — interpolation parameter values in [0, 1].
        logits_del:     np.ndarray, shape [S] — target-class logit along the deletion path.
        logits_ins:     np.ndarray, shape [S] — target-class logit along the insertion path.
        auc_del:        float — area under the deletion curve (lower = concept more important).
        auc_ins:        float — area under the insertion-gain curve (higher = concept more important).
    """
    concept_name: str
    class_index: int
    t_grid: np.ndarray
    logits_del: np.ndarray
    logits_ins: np.ndarray
    auc_del: float
    auc_ins: float


@dataclass
class LocalizationResult:
    """Results from concept map localization evaluation against GT mask.

    Attributes:
        concept_name:        str — concept being evaluated.
        layer_name:          str — layer at which the concept map was computed.
        soft_iou:            float — continuous Jaccard index in [0, 1] (higher is better).
        balanced_mse:        float — MSE balanced between inside/outside GT (lower is better).
        spatial_auc:         float — AUC-ROC over spatial cells (higher is better).
        neg_mean_activation: float — mean concept map value on negative images (lower is better).
    """
    concept_name: str
    layer_name: str
    soft_iou: float
    balanced_mse: float
    spatial_auc: float = 0.5
    neg_mean_activation: float = 0.0


@dataclass
class DistributionResult:
    """Results from distribution-based saliency comparison (Miró-Nicolau metrics).

    Attributes:
        concept_name: str — concept being evaluated.
        layer_name:   str — layer at which the concept map was computed.
        emd:          float — Earth Mover's Distance (lower is better).
        sim_min:      float — Similarity metric / histogram intersection (higher is better).
    """
    concept_name: str
    layer_name: str
    emd: float
    sim_min: float

# =========
# C-Insertion / C-Deletion 
# =========
def compute_concept_aligned_component(
    feature_maps: torch.Tensor,
    pooled_cav: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Extract the concept-aligned component D^c from feature maps.

    Implements Equations (7)-(8) from Visual-TCAV TMLR Appendix K:

        alpha_{i,j} = F_{i,j,:}^T p^c / ||p^c||^2
        D^c_{i,j,:} = ReLU(alpha_{i,j} - beta) * p^c

    Args:
        feature_maps: Tensor [N, C, H, W] — activations at the bottleneck layer.
        pooled_cav:   Tensor [C] — the pooled-CAV direction vector.
        beta:         float — baseline projection coefficient (mean alpha over negative set).

    Returns:
        D_concept: Tensor [N, C, H, W] — concept-aligned component at each location.
    """
    norm_sq = torch.dot(pooled_cav, pooled_cav)
    alpha = torch.einsum('nchw, c -> nhw', feature_maps, pooled_cav) / (norm_sq + 1e-10)
    excess = F.relu(alpha - beta)
    
    return excess[:, None, :, :] * pooled_cav[None, :, None, None]


def compute_beta_from_negatives(
    negative_feature_maps: torch.Tensor,
    pooled_cav: torch.Tensor,
) -> float:
    """Compute the baseline projection coefficient beta from negative examples.

    beta = mean over all spatial locations and negative images of alpha_{i,j}.

    Args:
        negative_feature_maps: Tensor [N_neg, C, H, W] — activations for random images.
        pooled_cav:            Tensor [C] — the pooled-CAV direction vector.

    Returns:
        beta: float — mean projection coefficient on the negative set.
    """
    norm_sq = torch.dot(pooled_cav, pooled_cav)
    alpha_neg = torch.einsum(
        'nchw, c -> nhw', negative_feature_maps, pooled_cav
    ) / (norm_sq + 1e-10)
    return alpha_neg.mean().item()


def c_insertion_deletion(
    feature_maps: torch.Tensor,
    pooled_cav: torch.Tensor,
    beta: float,
    model_wrapper,
    layer_name: str,
    class_index: int,
    concept_name: str = "",
    n_steps: int = 10,
    apply_relu: bool = True,
) -> CIDResult:
    """Run C-Insertion and C-Deletion for one concept on a batch of images.

    Implements Equation (9) from Visual-TCAV TMLR Appendix K:

        F^del(t) = F - t * D^c          (deletion: erase concept progressively)
        F^ins(t) = F - (1-t) * D^c      (insertion: restore concept progressively)

    Metrics (Equation 10):
        AUC_del = integral_0^1 z_k^del(t) dt
        AUC_ins = integral_0^1 [z_k^ins(t) - z_k^ins(0)] dt

    Args:
        feature_maps:  Tensor [N, C, H, W] — activations at the target layer.
        pooled_cav:    Tensor [C] — the pooled-CAV direction.
        beta:          float — from compute_beta_from_negatives().
        model_wrapper: PytorchModelWrapper instance (needs get_logits method).
        layer_name:    str — name of the target layer.
        class_index:   int — target class for logit tracking.
        concept_name:  str — for labeling the result.
        n_steps:       int — number of interpolation steps (default 10).
        apply_relu:    bool — clamp modified feature maps to non-negative.

    Returns:
        CIDResult with deletion/insertion curves and AUC values.
    """
    device = feature_maps.device
    D_c = compute_concept_aligned_component(feature_maps, pooled_cav, beta)
    t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    logits_del_list, logits_ins_list = [], []
    with torch.no_grad():
        for t in t_grid:
            F_del = feature_maps - t * D_c
            if apply_relu:
                F_del = F.relu(F_del)
            logits_del_list.append(model_wrapper.get_logits(F_del, layer_name)[:, class_index].mean().item())

            F_ins = feature_maps - (1.0 - t) * D_c
            if apply_relu:
                F_ins = F.relu(F_ins)
            logits_ins_list.append(model_wrapper.get_logits(F_ins, layer_name)[:, class_index].mean().item())

    t_np = t_grid.cpu().numpy()
    del_np = np.array(logits_del_list)
    ins_np = np.array(logits_ins_list)

    return CIDResult(
        concept_name=concept_name,
        class_index=class_index,
        t_grid=t_np,
        logits_del=del_np,
        logits_ins=ins_np,
        auc_del=float(np.trapz(del_np, t_np)),
        auc_ins=float(np.trapz(ins_np - ins_np[0], t_np)),
    )


def c_insertion_deletion_from_vtcav(
    vtcav_instance,
    concept_name: str,
    layer_name: str,
    class_index: int,
    test_feature_maps: torch.Tensor,
    random_feature_maps: Optional[torch.Tensor] = None,
    n_steps: int = 10,
    apply_relu: bool = True,
) -> CIDResult:
    """Convenience wrapper: run C-Insertion/C-Deletion using a VisualTCAV instance.

    Args:
        vtcav_instance:      A VisualTCAV instance with computed CAVs.
        concept_name:        str — the concept to evaluate.
        layer_name:          str — the layer to evaluate.
        class_index:         int — the target class index.
        test_feature_maps:   Tensor [N, C, H, W] — test activations.
        random_feature_maps: Tensor [N_neg, C, H, W] — negative activations (optional).
        n_steps:             int — number of interpolation steps.
        apply_relu:          bool — clamp to non-negative after intervention.

    Returns:
        CIDResult.
    """
    pooled_cav = vtcav_instance.computations[layer_name][concept_name].cav.direction

    if random_feature_maps is None:
        random_feature_maps = vtcav_instance._compute_random_activations(
            cache=True, layer_name=layer_name
        )
        if not isinstance(random_feature_maps, torch.Tensor):
            random_feature_maps = torch.tensor(
                random_feature_maps, dtype=torch.float32, device=vtcav_instance.device
            )

    beta = compute_beta_from_negatives(random_feature_maps, pooled_cav)
    return c_insertion_deletion(
        feature_maps=test_feature_maps,
        pooled_cav=pooled_cav,
        beta=beta,
        model_wrapper=vtcav_instance.model.model_wrapper,
        layer_name=layer_name,
        class_index=class_index,
        concept_name=concept_name,
        n_steps=n_steps,
        apply_relu=apply_relu,
    )


# ====
# Spatial location metrics for concept maps (Appendix H, De Santis et al. TMLR)
# ====

def soft_iou(concept_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Soft IoU (continuous Jaccard) between a concept map and GT mask.

    Soft-IoU = sum(M_hat * GT) / (sum(M_hat) + sum(GT) - sum(M_hat * GT))

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float in [0, 1] — 1.0 is perfect overlap.
    """
    intersection = (concept_map * gt_mask).sum()
    union = concept_map.sum() + gt_mask.sum() - intersection
    return (intersection / (union + 1e-10)).item()


def balanced_mse(concept_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Balanced MSE between concept map and GT mask (Visual-TCAV TMLR Appendix H).

    BMSE = 0.5 * mean((M - GT)^2 | inside GT) + 0.5 * mean((M - GT)^2 | outside GT)

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float >= 0 — lower is better.
    """
    sq_err = (concept_map - gt_mask) ** 2
    n_pos, n_neg = gt_mask.sum(), (1 - gt_mask).sum()
    mse_inside  = (sq_err * gt_mask).sum()       / (n_pos + 1e-10)
    mse_outside = (sq_err * (1 - gt_mask)).sum() / (n_neg + 1e-10)
    return (0.5 * mse_inside + 0.5 * mse_outside).item()


def spatial_auc(concept_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """AUC-ROC between concept map activations and GT binary mask.

    Treats each spatial cell as an independent sample. 
    - AUC = 0.5 means chance.
    - AUC = 1.0 means perfect discrimination.

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float in [0.5, 1.0]. Returns 0.5 if GT is all-zero or all-one.
    """
    scores = concept_map.detach().cpu().numpy().flatten()
    labels = gt_mask.detach().cpu().numpy().flatten()
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels, scores))


def evaluate_concept_map_localization(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
    concept_name: str = "",
    layer_name: str = "",
) -> LocalizationResult:
    """Evaluate a single concept map against a GT mask.

    The GT mask must be at the feature map resolution of the target layer.

    Args:
        concept_map:  Tensor [H, W] in [0, 1].
        gt_mask:      Tensor [H, W] in {0, 1}, same spatial resolution.
        concept_name: str — for labeling.
        layer_name:   str — for labeling.

    Returns:
        LocalizationResult.
    """
    assert concept_map.shape == gt_mask.shape, (
        f"Shape mismatch: concept_map {concept_map.shape} vs gt_mask {gt_mask.shape}. "
        "Downsample your GT mask to the feature map resolution first."
    )
    gt_float = gt_mask.float()
    return LocalizationResult(
        concept_name=concept_name,
        layer_name=layer_name,
        soft_iou=soft_iou(concept_map, gt_float),
        balanced_mse=balanced_mse(concept_map, gt_float),
        spatial_auc=spatial_auc(concept_map, gt_float),
    )


def evaluate_concept_map_batch(
    vtcav_instance,
    concept_name: str,
    layer_name: str,
    test_feature_maps: torch.Tensor,
    gt_masks: torch.Tensor,
    negative_feature_maps: Optional[torch.Tensor] = None,
) -> LocalizationResult:
    """Evaluate concept map localization over a batch, returning mean metrics.

    Args:
        vtcav_instance:        VisualTCAV instance with computed CAVs.
        concept_name:          str — the concept.
        layer_name:            str — the layer.
        test_feature_maps:     Tensor [N, C, H, W] — positive activations.
        gt_masks:              Tensor [N, H_fm, W_fm] — binary GT masks.
        negative_feature_maps: Optional Tensor [N_neg, C, H, W] — negative activations.

    Returns:
        LocalizationResult with metrics averaged over the batch.
    """
    concept_layer = vtcav_instance.computations[layer_name][concept_name]
    direction = concept_layer.cav.direction
    emblem    = concept_layer.cav.concept_emblem

    siou_list, bmse_list, sauc_list = [], [], []
    for i in range(test_feature_maps.shape[0]):
        cmap = vtcav_instance._compute_concept_map(test_feature_maps[i], direction, emblem)
        gt_i = gt_masks[i].float().to(cmap.device)
        if cmap.shape != gt_i.shape:
            gt_i = F.interpolate(
                gt_i[None, None], size=cmap.shape, mode='bilinear', align_corners=False
            ).squeeze()
            gt_i = (gt_i > 0.5).float()
        siou_list.append(soft_iou(cmap, gt_i))
        bmse_list.append(balanced_mse(cmap, gt_i))
        sauc_list.append(spatial_auc(cmap, gt_i))

    neg_mean_act = 0.0
    if negative_feature_maps is not None:
        neg_acts = [
            vtcav_instance._compute_concept_map(
                negative_feature_maps[i], direction, emblem
            ).mean().item()
            for i in range(negative_feature_maps.shape[0])
        ]
        neg_mean_act = float(np.mean(neg_acts))

    return LocalizationResult(
        concept_name=concept_name,
        layer_name=layer_name,
        soft_iou=float(np.mean(siou_list)),
        balanced_mse=float(np.mean(bmse_list)),
        spatial_auc=float(np.mean(sauc_list)),
        neg_mean_activation=neg_mean_act,
    )

# =========
# Distribution Metrics
# =========

def _to_distribution(tensor: torch.Tensor) -> np.ndarray:
    """Normalize a 2D tensor to a probability distribution (flattened, sums to 1)."""
    flat = tensor.detach().cpu().numpy().flatten().astype(np.float64)
    flat = np.maximum(flat, 0.0)
    total = flat.sum()
    return flat / total if total > 0 else np.ones_like(flat) / len(flat)


def earth_movers_distance(concept_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Wasserstein-1 distance between concept map and GT mask distributions.

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float >= 0 — lower is better.
    """
    return float(wasserstein_distance(_to_distribution(concept_map), _to_distribution(gt_mask)))


def similarity_metric_min(concept_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Histogram intersection (MIN) similarity metric from Judd et al. (2012).

    MIN(SM, GT) = sum_i min(SM_i, GT_i) where both are normalized distributions.

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float in [0, 1] — higher is better.
    """
    return float(np.minimum(_to_distribution(concept_map), _to_distribution(gt_mask)).sum())


def evaluate_distribution_metrics(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
    concept_name: str = "",
    layer_name: str = "",
) -> DistributionResult:
    """Compute EMD and MIN similarity for one concept map vs GT.

    Args:
        concept_map:  Tensor [H, W] in [0, 1].
        gt_mask:      Tensor [H, W] in {0, 1}, same resolution.
        concept_name: str — for labeling.
        layer_name:   str — for labeling.

    Returns:
        DistributionResult.
    """
    return DistributionResult(
        concept_name=concept_name,
        layer_name=layer_name,
        emd=earth_movers_distance(concept_map, gt_mask),
        sim_min=similarity_metric_min(concept_map, gt_mask),
    )

# =========
# Aggregation & Reporting
# =========

def rank_concepts_by_cid(results: list[CIDResult]) -> dict:
    """Rank concepts by C-Insertion/C-Deletion metrics and compute Spearman rho.

    Args:
        results: List of CIDResult, one per concept, in attribution-rank order.

    Returns:
        dict with attribution_rank, auc_del_rank, auc_ins_rank, spearman_rho_del.
    """
    from scipy.stats import spearmanr

    n = len(results)
    attr_rank = list(range(n))
    del_sorted = sorted(range(n), key=lambda i: results[i].auc_del)
    del_ranks = [0] * n
    for rank, idx in enumerate(del_sorted):
        del_ranks[idx] = rank
    rho, _ = spearmanr(attr_rank, del_ranks)

    return {
        'attribution_rank': [r.concept_name for r in results],
        'auc_del_rank':     [results[i].concept_name for i in del_sorted],
        'auc_ins_rank':     [
            results[i].concept_name
            for i in sorted(range(n), key=lambda i: results[i].auc_ins, reverse=True)
        ],
        'spearman_rho_del': float(rho) if not np.isnan(rho) else 0.0,
    }


def print_cid_summary(results: list[CIDResult]):
    """Print a formatted summary table of C-Insertion/C-Deletion results."""
    print(f"\n{'Concept':<20} {'AUC_del ↓':>12} {'AUC_ins ↑':>12}")
    print("-" * 46)
    for r in results:
        print(f"{r.concept_name:<20} {r.auc_del:>12.4f} {r.auc_ins:>12.4f}")
    if len(results) > 1:
        ranking = rank_concepts_by_cid(results)
        print(f"\nSpearman ρ (attribution vs AUC_del): {ranking['spearman_rho_del']:.4f}")
        print(f"AUC_del rank:  {' > '.join(ranking['auc_del_rank'])}")
        print(f"AUC_ins rank:  {' > '.join(ranking['auc_ins_rank'])}")


def print_localization_summary(results: list[LocalizationResult]):
    """Print a formatted summary table of localization results."""
    print(f"\n{'Concept':<15} {'Layer':<20} {'Soft IoU ↑':>12} {'Bal. MSE ↓':>12} {'AUC ↑':>10} {'Neg. Act. ↓':>12}")
    print("-" * 85)
    for r in results:
        print(
            f"{r.concept_name:<15} {r.layer_name:<20} "
            f"{r.soft_iou:>12.4f} {r.balanced_mse:>12.4f} "
            f"{r.spatial_auc:>10.4f} {r.neg_mean_activation:>12.4f}"
        )
