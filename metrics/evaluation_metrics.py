#####
#   Visual-TCAV Faithfulness Evaluation Metrics
#
#   Implements three families of objective metrics for assessing the fidelity
#   of Visual-TCAV concept attributions and concept maps:
#
#   1. C-Insertion / C-Deletion (Appendix K, De Santis et al. TMLR):
#      Causal intervention in feature-map space. Progressively erases or
#      re-inserts the concept-aligned component and records the target logit.
#      Metric: AUC of the logit curve.
#
#   2. Concept Map Localization (Appendix H, De Santis et al. TMLR):
#      Compares normalized concept maps against binary GT masks using
#      Soft IoU (continuous Jaccard) and Balanced MSE.
#
#   3. Saliency Distribution Metrics (Miró-Nicolau et al., AIJ 2024):
#      Earth Mover's Distance (EMD) and Similarity Metric (MIN) treating
#      both saliency map and GT as probability distributions.
#
#   Integration point: These functions operate on the intermediate tensors
#   produced by VisualTCAV._compute_concept_map(), VisualTCAV._compute_cavs(),
#   and PytorchModelWrapper.get_logits(). They do NOT modify the model.
#####


import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score


# ============================================================================
# Data classes for structured results
# ============================================================================

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
        concept_name:       str — concept being evaluated.
        layer_name:         str — layer at which the concept map was computed.
        soft_iou:           float — continuous Jaccard index in [0, 1] (higher is better).
        balanced_mse:       float — MSE balanced between inside/outside GT (lower is better).
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


# ============================================================================
# 1. C-Insertion / C-Deletion
# ============================================================================

def compute_concept_aligned_component(
    feature_maps: torch.Tensor,
    pooled_cav: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Extract the concept-aligned component D^c from feature maps.

    Implements Equations (7)-(8) from Visual-TCAV TMLR Appendix K:

        alpha_{i,j} = F_{i,j,:}^T p^c / ||p^c||^2
        D^c_{i,j,:} = ReLU(alpha_{i,j} - beta) * p^c

    Only the *excess* positive alignment relative to the negative-set
    baseline beta is considered concept-aligned.

    Args:
        feature_maps: Tensor [N, C, H, W] — activations at the bottleneck layer.
        pooled_cav:   Tensor [C] — the pooled-CAV direction vector.
        beta:         float — baseline projection coefficient, computed as
                      mean(alpha_{i,j}) over all spatial locations and all
                      images in the negative (random) set D^c_neg.

    Returns:
        D_concept: Tensor [N, C, H, W] — concept-aligned component at each location.
    """
    # Projection coefficient at each spatial location
    # alpha_{i,j} = F_{i,j,:}^T p^c / ||p^c||^2
    norm_sq = torch.dot(pooled_cav, pooled_cav)  # scalar
    alpha = torch.einsum('nchw, c -> nhw', feature_maps, pooled_cav) / (norm_sq + 1e-10)
    # alpha: [N, H, W]

    # Excess alignment beyond negative baseline, clamped to non-negative
    excess = F.relu(alpha - beta)  # [N, H, W]

    # Concept-aligned component: broadcast excess * p^c
    D_concept = excess[:, None, :, :] * pooled_cav[None, :, None, None]
    # D_concept: [N, C, H, W]

    return D_concept


def compute_beta_from_negatives(
    negative_feature_maps: torch.Tensor,
    pooled_cav: torch.Tensor,
) -> float:
    """Compute the baseline projection coefficient beta from negative examples.

    beta = mean over all spatial locations and all negative images of:
        alpha_{i,j} = F_{i,j,:}^T p^c / ||p^c||^2

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

    At each step t in {0, 0.1, ..., 1.0}, we forward-pass through the model
    from the bottleneck layer and record the target-class logit.

    The metrics are (Equation 10):

        AUC_del = integral_0^1 z_k^del(t) dt
        AUC_ins = integral_0^1 [z_k^ins(t) - z_k^ins(0)] dt

    Args:
        feature_maps:  Tensor [N, C, H, W] — activations at the target layer.
        pooled_cav:    Tensor [C] — the pooled-CAV direction.
        beta:          float — baseline from compute_beta_from_negatives().
        model_wrapper: PytorchModelWrapper instance (needs get_logits method).
        layer_name:    str — name of the target layer.
        class_index:   int — target class for logit tracking.
        concept_name:  str — for labeling the result.
        n_steps:       int — number of interpolation steps (default 10 → step size 0.1).
        apply_relu:    bool — if True, clamp modified feature maps to non-negative
                       (required for post-ReLU layers).

    Returns:
        CIDResult with deletion/insertion curves and AUC values.
    """
    device = feature_maps.device

    # Concept-aligned component: [N, C, H, W]
    D_c = compute_concept_aligned_component(feature_maps, pooled_cav, beta)

    t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    logits_del_list = []
    logits_ins_list = []

    with torch.no_grad():
        for t in t_grid:
            # --- Deletion path: F^del(t) = F - t * D^c ---
            F_del = feature_maps - t * D_c
            if apply_relu:
                F_del = F.relu(F_del)
            logits_del = model_wrapper.get_logits(F_del, layer_name)
            logits_del_list.append(logits_del[:, class_index].mean().item())

            # --- Insertion path: F^ins(t) = F - (1-t) * D^c ---
            F_ins = feature_maps - (1.0 - t) * D_c
            if apply_relu:
                F_ins = F.relu(F_ins)
            logits_ins = model_wrapper.get_logits(F_ins, layer_name)
            logits_ins_list.append(logits_ins[:, class_index].mean().item())

    t_np = t_grid.cpu().numpy()
    del_np = np.array(logits_del_list)
    ins_np = np.array(logits_ins_list)

    # AUC via trapezoidal rule (Equation 10)
    auc_del = float(np.trapz(del_np, t_np))
    auc_ins = float(np.trapz(ins_np - ins_np[0], t_np))

    return CIDResult(
        concept_name=concept_name,
        class_index=class_index,
        t_grid=t_np,
        logits_del=del_np,
        logits_ins=ins_np,
        auc_del=auc_del,
        auc_ins=auc_ins,
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

    Extracts the pooled-CAV and computes beta from the VisualTCAV's cached
    computations and random activations, then delegates to c_insertion_deletion().

    Args:
        vtcav_instance:    A VisualTCAV (Local or Global) instance with computed CAVs.
        concept_name:      str — the concept to evaluate.
        layer_name:        str — the layer to evaluate.
        class_index:       int — the target class index.
        test_feature_maps: Tensor [N, C, H, W] — test images' activations at this layer.
        random_feature_maps: Tensor [N_neg, C, H, W] — random/negative activations.
                           If None, recomputed from vtcav_instance.
        n_steps:           int — number of interpolation steps.
        apply_relu:        bool — clamp to non-negative after intervention.

    Returns:
        CIDResult.
    """
    # Extract the pooled-CAV direction from the cached computation
    concept_layer = vtcav_instance.computations[layer_name][concept_name]
    pooled_cav = concept_layer.cav.direction  # Tensor [C]

    # Compute beta from negative examples
    if random_feature_maps is None:
        random_feature_maps = vtcav_instance._compute_random_activations(
            cache=True, layer_name=layer_name
        )
        if not isinstance(random_feature_maps, torch.Tensor):
            random_feature_maps = torch.tensor(
                random_feature_maps, dtype=torch.float32,
                device=vtcav_instance.device
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


# ============================================================================
# 2. Concept Map Localization Metrics (Appendix H)
# ============================================================================

def soft_iou(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """Soft IoU (continuous Jaccard index) between a concept map and GT mask.

    This is the threshold-free generalization of IoU used in Appendix H
    (De Santis et al. TMLR), originally from Rahman & Wang (2016):

        Soft-IoU = sum(M_hat * GT) / (sum(M_hat) + sum(GT) - sum(M_hat * GT))

    where M_hat is the normalized concept map in [0,1] and GT is the binary mask.

    Args:
        concept_map: Tensor [H, W] in [0, 1] — normalized concept map from
                     VisualTCAV._compute_concept_map().
        gt_mask:     Tensor [H, W] in {0, 1} — binary ground-truth mask at the
                     same spatial resolution as the concept map (i.e., at the
                     feature map resolution, NOT the input image resolution).

    Returns:
        float in [0, 1] — 1.0 is perfect overlap.
    """
    intersection = (concept_map * gt_mask).sum()
    union = concept_map.sum() + gt_mask.sum() - intersection
    return (intersection / (union + 1e-10)).item()


def balanced_mse(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """Balanced MSE between concept map and GT mask.

    Balances the error contribution from inside and outside the GT region
    equally, preventing large background areas from dominating:

        BMSE = 0.5 * mean((M - GT)^2 | inside GT) + 0.5 * mean((M - GT)^2 | outside GT)

    This is the metric used in Visual-TCAV TMLR Appendix H, Table 3.

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float >= 0 — lower is better.
    """
    sq_err = (concept_map - gt_mask) ** 2  # [H, W]

    n_pos = gt_mask.sum()
    n_neg = (1 - gt_mask).sum()

    # Mean squared error inside GT region
    mse_inside = (sq_err * gt_mask).sum() / (n_pos + 1e-10)
    # Mean squared error outside GT region
    mse_outside = (sq_err * (1 - gt_mask)).sum() / (n_neg + 1e-10)

    return (0.5 * mse_inside + 0.5 * mse_outside).item()


def spatial_auc(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """AUC-ROC between concept map activations and GT binary mask.

    Treats each spatial cell in the feature map as an independent sample:
      - score  = concept map activation at that cell (continuous, in [0, 1])
      - label  = 1 if the GT shape is present at that cell, 0 otherwise

    AUC-ROC measures how well the concept map discriminates shape-present cells
    from shape-absent cells, independently of any threshold, scale, or bounding-box
    extraction. AUC = 0.5 means the concept map is at chance; AUC = 1.0 means
    it perfectly separates the two populations.

    Args:
        concept_map: Tensor [H, W] in [0, 1] from _compute_concept_map().
        gt_mask:     Tensor [H, W] in {0, 1} at feature map resolution.

    Returns:
        float in [0.5, 1.0]. Returns 0.5 (chance) if GT is all-zero or all-one.
    """
    scores = concept_map.detach().cpu().numpy().flatten()
    labels = gt_mask.detach().cpu().numpy().flatten()

    # AUC is undefined if GT is all-zero or all-one
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5

    return float(roc_auc_score(labels, scores))


def evaluate_concept_map_localization(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
    concept_name: str = "",
    layer_name: str = "",
) -> LocalizationResult:
    """Evaluate a single concept map against a GT mask using both localization metrics.

    The GT mask must be at the *feature map resolution* of the target layer,
    not at the input image resolution. Downsample your input-space GT mask to
    match the spatial dimensions of the feature maps before calling this.

    Example:
        # If your GT is at input resolution [224, 224]:
        gt_input = torch.tensor(binary_mask)  # [224, 224]
        # And your feature maps at the target layer are [7, 7]:
        gt_downsampled = F.interpolate(
            gt_input[None, None].float(), size=(7, 7), mode='bilinear'
        ).squeeze() > 0.5  # threshold back to binary

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}, same spatial resolution.
        concept_name: str — for labeling.
        layer_name:   str — for labeling.

    Returns:
        LocalizationResult.
    """
    assert concept_map.shape == gt_mask.shape, (
        f"Shape mismatch: concept_map {concept_map.shape} vs gt_mask {gt_mask.shape}. "
        f"Downsample your GT mask to the feature map resolution first."
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
    """Evaluate concept map localization over a batch of images.

    Computes concept maps for each test image, evaluates against corresponding
    GT masks, and returns the mean metrics. Also computes mean activation on
    negative images (if provided) as a false-positive rate indicator.

    Args:
        vtcav_instance:    VisualTCAV instance with computed CAVs.
        concept_name:      str — the concept.
        layer_name:        str — the layer.
        test_feature_maps: Tensor [N, C, H, W] — positive test images' activations.
        gt_masks:          Tensor [N, H_fm, W_fm] — binary GT masks at feature map resolution.
        negative_feature_maps: Optional Tensor [N_neg, C, H, W] — images where the concept
                              is absent, used to compute neg_mean_activation.

    Returns:
        LocalizationResult with metrics averaged over the batch.
    """
    concept_layer = vtcav_instance.computations[layer_name][concept_name]
    direction = concept_layer.cav.direction
    emblem = concept_layer.cav.concept_emblem

    siou_list = []
    bmse_list = []
    sauc_list = []

    for i in range(test_feature_maps.shape[0]):
        cmap = vtcav_instance._compute_concept_map(
            test_feature_maps[i], direction, emblem
        )
        gt_i = gt_masks[i].float().to(cmap.device)

        # Ensure spatial dimensions match
        if cmap.shape != gt_i.shape:
            gt_i = F.interpolate(
                gt_i[None, None], size=cmap.shape, mode='bilinear', align_corners=False
            ).squeeze()
            gt_i = (gt_i > 0.5).float()

        siou_list.append(soft_iou(cmap, gt_i))
        bmse_list.append(balanced_mse(cmap, gt_i))
        sauc_list.append(spatial_auc(cmap, gt_i))

    # Negative activation (false-positive rate)
    neg_mean_act = 0.0
    if negative_feature_maps is not None:
        neg_acts = []
        for i in range(negative_feature_maps.shape[0]):
            cmap_neg = vtcav_instance._compute_concept_map(
                negative_feature_maps[i], direction, emblem
            )
            neg_acts.append(cmap_neg.mean().item())
        neg_mean_act = float(np.mean(neg_acts))

    return LocalizationResult(
        concept_name=concept_name,
        layer_name=layer_name,
        soft_iou=float(np.mean(siou_list)),
        balanced_mse=float(np.mean(bmse_list)),
        spatial_auc=float(np.mean(sauc_list)),
        neg_mean_activation=neg_mean_act,
    )


# ============================================================================
# 3. Distribution Metrics (Miró-Nicolau et al., AIJ 2024)
# ============================================================================

def _to_distribution(tensor: torch.Tensor) -> np.ndarray:
    """Normalize a 2D tensor to a valid probability distribution.

    Flattens, clamps to non-negative, and normalizes to sum to 1.
    If the tensor is all zeros, returns a uniform distribution.

    Args:
        tensor: Tensor [H, W].

    Returns:
        np.ndarray [H*W] summing to 1.
    """
    flat = tensor.detach().cpu().numpy().flatten().astype(np.float64)
    flat = np.maximum(flat, 0.0)
    total = flat.sum()
    if total > 0:
        return flat / total
    else:
        return np.ones_like(flat) / len(flat)


def earth_movers_distance(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """Earth Mover's Distance (Wasserstein-1) between concept map and GT mask.

    Both are treated as 1D probability distributions (flattened). This is the
    metric used in Miró-Nicolau et al. (2024), Equation (25).

    Note: For 2D spatial maps, using the 1D Wasserstein distance on flattened
    arrays is an approximation. For exact 2D EMD you'd need POT or similar
    optimal-transport solvers, which is substantially more expensive. The 1D
    version is what Miró-Nicolau et al. use in their implementation.

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float >= 0 — lower is better (0 = identical distributions).
    """
    p = _to_distribution(concept_map)
    q = _to_distribution(gt_mask)
    return float(wasserstein_distance(p, q))


def similarity_metric_min(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """Similarity Metric (MIN / histogram intersection) from Judd et al. (2012).

    Used in Miró-Nicolau et al. (2024), Equation (26):

        MIN(SM, GT) = sum_i min(SM_i, GT_i)

    where both SM and GT are normalized to probability distributions.

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}.

    Returns:
        float in [0, 1] — higher is better (1 = identical distributions).
    """
    p = _to_distribution(concept_map)
    q = _to_distribution(gt_mask)
    return float(np.minimum(p, q).sum())


def evaluate_distribution_metrics(
    concept_map: torch.Tensor,
    gt_mask: torch.Tensor,
    concept_name: str = "",
    layer_name: str = "",
) -> DistributionResult:
    """Compute both distribution-based metrics for one concept map vs GT.

    Args:
        concept_map: Tensor [H, W] in [0, 1].
        gt_mask:     Tensor [H, W] in {0, 1}, same resolution.
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


# ============================================================================
# Aggregation & Reporting
# ============================================================================

def rank_concepts_by_cid(
    results: list[CIDResult],
) -> dict:
    """Rank concepts by C-Insertion/C-Deletion metrics and compute Spearman rho.

    Given a list of CIDResult (one per concept), ranks concepts by their
    Visual-TCAV attribution (assumed to be in the order they were passed)
    and computes Spearman rank correlation with AUC_del ranking.

    This implements the validation from Visual-TCAV TMLR Table 8.

    Args:
        results: List of CIDResult, one per concept. The list order is assumed
                 to be the attribution-rank order (highest attribution first).

    Returns:
        dict with keys:
            'attribution_rank': list of concept names in attribution order.
            'auc_del_rank': list of concept names sorted by AUC_del (ascending).
            'auc_ins_rank': list of concept names sorted by AUC_ins (descending).
            'spearman_rho_del': Spearman rank correlation between attribution
                                and AUC_del rankings.
    """
    from scipy.stats import spearmanr

    n = len(results)

    # Attribution rank (as given — index 0 = highest attribution)
    attr_rank = list(range(n))

    # AUC_del rank: lower AUC_del = more important = rank 0
    del_sorted_indices = sorted(range(n), key=lambda i: results[i].auc_del)
    del_ranks = [0] * n
    for rank, idx in enumerate(del_sorted_indices):
        del_ranks[idx] = rank

    rho, _ = spearmanr(attr_rank, del_ranks)

    return {
        'attribution_rank': [r.concept_name for r in results],
        'auc_del_rank': [results[i].concept_name for i in del_sorted_indices],
        'auc_ins_rank': [
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
