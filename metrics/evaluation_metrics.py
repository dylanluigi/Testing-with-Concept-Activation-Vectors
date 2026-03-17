# Backward-compat shim — see tools/metrics.py
from tools.metrics import *
from tools.metrics import (
    CIDResult, LocalizationResult, DistributionResult,
    aggregate_tcav_runs,
    compute_concept_aligned_component, compute_beta_from_negatives,
    c_insertion_deletion, c_insertion_deletion_from_vtcav,
    soft_iou, balanced_mse, spatial_auc,
    evaluate_concept_map_localization, evaluate_concept_map_batch,
    earth_movers_distance, similarity_metric_min, evaluate_distribution_metrics,
    rank_concepts_by_cid, print_cid_summary, print_localization_summary,
)
