#####
#   VisualTCAV — PyTorch + Captum Implementation
#
#   Port of the TensorFlow VisualTCAV (De Santis et al.) to PyTorch,
#   leveraging Captum's attribution API to replace manual hook/gradient logic.
#
#   Architecture decision: We use Captum as *infrastructure* (activation extraction,
#   integrated gradients) but keep the Visual-TCAV-specific logic (centroid-based CAVs,
#   concept emblems, spatial concept maps, masked attributions) as custom code.
#
#   Captum's own TCAV implementation (captum.concept.TCAV) uses classifier-based CAVs
#   (linear SVM), which is a fundamentally different approach from Visual-TCAV's
#   centroid-based direction. We deliberately do NOT use Captum's TCAV class.
#   See the CAV_METHOD parameter in VisualTCAV.__init__ for switching between approaches.
#####


#####
# Imports
#####

import sys
sys.dont_write_bytecode = True

import os
import numpy as np
from numpy.linalg import norm
from joblib import dump, load
import PIL.Image, PIL.ImageFilter
from tqdm import tqdm
from multiprocessing import dummy as multiprocessing
# from prettytable import PrettyTable
import pandas as pd

from matplotlib import pyplot as plt, cm as cm
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# Captum — used for two specific purposes:
#   1. LayerActivation: extracts intermediate feature maps without manual forward hooks.
#      This replaces the TF pattern of Model(inputs=..., outputs=layer.output) slicing.
#   2. LayerIntegratedGradients: computes layer-level integrated gradients in one call.
#      This replaces the manual _interpolate_images() + GradientTape loop from TF.
#   3. Concept + CustomIterableDataset: manages concept image loading and batching.
#      This replaces the custom ImageActivationGenerator._load_images_from_files() pipeline.
from captum.attr import LayerActivation, LayerIntegratedGradients
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

# Optional: sklearn for classifier-based CAVs (alternative to centroid-based)
# This is only imported when cav_method="classifier" is selected.

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#####
# Device configuration
#####

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####
# Utility functions
#####

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def nth_highest_index(arr, n):
    arr_np = np.array(arr)
    idx = np.argsort(-arr_np, kind='stable')[n - 1]
    return idx


def contraharmonic_mean(arr, dim=(2, 3)):
    arr = arr.float()
    numerator = torch.sum(arr ** 2, dim=dim)
    denominator = torch.sum(arr, dim=dim)
    return numerator / (denominator + 1e-7)


#####
# VisualTCAV base class
#####

class VisualTCAV():
    """Base class for Visual-TCAV explanations.

    This class manages the directory structure, model binding, concept/layer configuration,
    CAV computation and integrated gradients.

    CAV computation method:
        The `cav_method` parameter controls how CAV directions are computed:

        - "centroid" (default, original Visual-TCAV proposition in the paper):
            direction = mean(concept_activations) - mean(random_activations)
            This approach assumes linear separability via centroids, no accuracy metric. Having said
            that, there's no need for training, it's deterministic and fast, good for small
            concept sets.

        - "classifier" (Captum/standard TCAV approach):
            Trains a LinearSVC to separate concept vs random activations.
            direction = classifier.coef_ (the learned hyperplane normal).
            This approach provides accuracy metric, a better separation for more
            complex distributions, problematic because it has non-deterministic as it is
            dependent on the train/test split.

        Both methods produce a direction vector in activation space. The rest of the
        Visual-TCAV pipeline (concept maps, emblems, masked attributions) is identical
        regardless of which method generated the direction.
    """

    def __init__(
        self,
        model,
        visual_tcav_dir="VisualTCAV",
        clear_cache=False,
        batch_size=250,
        models_dir=None,
        cache_dir=None,
        test_images_dir=None,
        concept_images_dir=None,
        random_images_folder=None,
        cav_method="centroid",
        device=None,
    ):
        """Initialize the VisualTCAV base.

        Args:
            model: A Model instance wrapping a PyTorch nn.Module.
            visual_tcav_dir: Root directory for VisualTCAV file structure.
            clear_cache: If True, delete all cached files on init.
            batch_size: Batch size for activation extraction.
            models_dir: Override for models directory path.
            cache_dir: Override for cache directory path.
            test_images_dir: Override for test images directory path.
            concept_images_dir: Override for concept images directory path.
            random_images_folder: Name of the folder containing random/negative images.
            cav_method: "centroid" (Visual-TCAV original) or "classifier" (sklearn SVM).
                See class docstring for detailed comparison.
            device: torch.device to use. Defaults to CUDA if available.
        """
        # CAV method validation
        if cav_method not in ("centroid", "classifier"):
            raise ValueError(f"cav_method must be 'centroid' or 'classifier', got '{cav_method}'")
        
        self.cav_method = cav_method

        # Device
        self.device = device or DEVICE

        # Folders and directories
        self.models_dir = os.path.join(visual_tcav_dir, "models") if not models_dir else models_dir
        self.cache_base_dir = os.path.join(visual_tcav_dir, "cache") if not cache_dir else cache_dir
        self.cache_dir = self.cache_base_dir
        self.test_images_dir = os.path.join(visual_tcav_dir, "test_images") if not test_images_dir else test_images_dir
        self.concept_images_dir = os.path.join(visual_tcav_dir, "concept_images") if not concept_images_dir else concept_images_dir
        self.random_images_folder = "random" if not random_images_folder else random_images_folder

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_base_dir, exist_ok=True)
        os.makedirs(self.test_images_dir, exist_ok=True)
        os.makedirs(self.concept_images_dir, exist_ok=True)

        self.batch_size = batch_size

        # Model
        self.model = None
        if model:
            self._bindModel(model)

        if clear_cache:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))

        # Concepts/Layers attributes
        self.concepts = []
        self.layers = []

        # Computations
        self.computations = {}
        self.random_acts = {}

    def setConcepts(self, concept_names):
        """Set the list of concept names to analyze."""
        self.concepts = []
        for concept_name in concept_names:
            if concept_name not in self.concepts:
                self.concepts.append(concept_name)

    def setLayers(self, layer_names):
        """Set the list of layer names to analyze.

        In PyTorch, layer_names are dot-separated module paths (e.g., 'layer4.2.conv3')
        as returned by model.named_modules(). Unlike TF where layers have unique string
        names, PyTorch modules are addressed by their position in the module tree.
        """
        self.layers = []
        for layer_name in layer_names:
            if layer_name not in self.layers:
                self.layers.append(layer_name)

    def predict(self, no_sort=False):
        """Run prediction on the test image (LocalVisualTCAV only).

        Returns a Predictions object with sorted class predictions.
        """
        if not isinstance(self, LocalVisualTCAV):
            raise Exception("Please use a local explainer")
        if not self.model:
            raise Exception("Please instantiate a Model first")

        # Preprocess and predict
        preprocessed = self.model.preprocessing_function(self.resized_imgs)

        self.predictions = self.model.model_wrapper.get_predictions(preprocessed)

        # Sort & add class names
        self.predictions = np.array([
            self._sortTargetClasses(
                prediction,
                self.model.model_wrapper.id_to_label,
                no_sort
            ) for prediction in self.predictions
        ])

        # Return the classes
        return Predictions(self.predictions, self.test_image_filename, self.model.model_name)

    #####
    # Private methods
    #####

    def _bindModel(self, model):
        """Bind a Model instance: instantiate the model wrapper and activation generator."""
        # Folders and directories
        if isinstance(model.graph_path_filename, nn.Module):
            model.graph_path_dir = model.graph_path_filename
        else:
            model.graph_path_dir = os.path.join(self.models_dir, model.model_name, model.graph_path_filename)

        if isinstance(model.label_path_filename, list):
            model.label_path_dir = model.label_path_filename
        else:
            model.label_path_dir = os.path.join(self.models_dir, model.model_name, model.label_path_filename)

        # FIX: Only instantiate if model_wrapper is still a class (type)
        if isinstance(model.model_wrapper, type):
            model.model_wrapper = model.model_wrapper(
                model.graph_path_dir, model.label_path_dir, self.batch_size,
                device=self.device, image_shape=model.image_shape
            )

            model.activation_generator = model.activation_generator(
                model_wrapper=model.model_wrapper,
                concept_images_dir=self.concept_images_dir,
                cache_dir=self.cache_dir,
                preprocessing_function=model.preprocessing_function,
                max_examples=model.max_examples,
            )

        # Model's cache dir
        self.cache_dir = os.path.join(self.cache_base_dir, model.model_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Store the model
        self.model = model

    def _sortTargetClasses(self, predictions, id_to_label, no_sort=False):
        """Convert raw prediction array to sorted Prediction objects (top 10)."""
        indexed_arr = list(enumerate(predictions))
        sorted_arr = indexed_arr if no_sort else sorted(indexed_arr, key=lambda x: x[1], reverse=True)
        return [
            Prediction(
                class_index=sorted_element[0],
                class_name=id_to_label(sorted_element[0]),
                confidence=sorted_element[1],
            ) for i, sorted_element in enumerate(sorted_arr) if i < 10
        ]

    def _compute_integrated_gradients(self, feature_maps, layer_name, class_index):

        # TODO check this

        """Compute integrated gradients at a specific layer for a target class.

        This is where Captums API provides a big simplification. The TF version
        required:
            1. _interpolate_images(): manually create m_steps interpolated feature maps
            2. get_gradient_of_score(): GradientTape for each interpolation step
            3. Trapezoidal rule accumulation
        All of that is replaced by a single LayerIntegratedGradients.attribute() call.

        ALTERNATIVE APPROACH (manual, without Captum):
            If you wanted to avoid the Captum dependency for IG, you could implement it
            manually in PyTorch:

                baseline = torch.zeros_like(feature_maps)
                alphas = torch.linspace(0, 1, self.m_steps + 1)
                interpolated = baseline + alphas[:, None, None, None] * (feature_maps - baseline)
                interpolated.requires_grad_(True)
                logits = model_forward_from_layer(interpolated, layer_name)
                logits[:, class_index].sum().backward()
                grads = interpolated.grad  # [m_steps+1, C, H, W]
                ig = ((grads[:-1] + grads[1:]) / 2).mean(dim=0)

            However, this requires building a sub-model from the target layer to the
            output (equivalent to TF's Model(inputs=layer, outputs=logits)), managing
            retain_graph, and handling memory for m_steps forward/backward passes.
            Captum handles all of this internally, including proper memory cleanup.

        Args:
            feature_maps: Tensor [C, H, W] — activations at the target layer.
            layer_name: Name of the target layer.
            class_index: Target class index for IG computation.

        Returns:
            Integrated gradients tensor [C, H, W], same shape as feature_maps.
        """

        layer_module = self.model.model_wrapper.get_layer_module(layer_name)

        # Captum's LayerIntegratedGradients handles the entire IG pipeline:
        # interpolation, forward passes, gradient computation, and accumulation.
        # n_steps corresponds to m_steps in the TF version.
        lig = LayerIntegratedGradients(
            self.model.model_wrapper.model, layer_module
        )

        # We need the original input image to compute IG through the full model.
        # The feature_maps are the activations at the target layer, but Captum
        # needs the model input to run the full forward pass internally.
        #
        # Note: This is stored by the caller (explain()) before invoking this method.
        input_tensor = self._current_input_tensor


        target = 0 if self.model.binary_classification else class_index 
        
        ig_attr = lig.attribute(
            input_tensor,
            n_steps=self.m_steps,
            target=target,
        )

        # ig_attr shape: same as the layer output, e.g. [1, C, H, W]
        return ig_attr[0]  # Remove batch dim -> [C, H, W]

    def _compute_random_activations(self, cache, layer_name):
        """Load or compute feature maps for the random/negative concept images.

        Uses joblib caching to avoid recomputation across runs.
        """
        cache_path = os.path.join(
            self.cache_dir,
            f'rnd_acts_{self.model.max_examples}_{self.random_images_folder}_{layer_name}.joblib'
        )
        if cache and os.path.isfile(cache_path):
            random_acts = load(cache_path)
        else:
            random_acts = self._compute_random(layer_name)
            if cache:
                dump(random_acts, cache_path, compress=3)
        return random_acts

    def _compute_random(self, layer_name):
        """Get feature maps for random images at the given layer."""
        return self.model.activation_generator.get_feature_maps_for_concept(
            self.random_images_folder, layer_name
        )

    def _compute_cavs(self, cache, concept_name, layer_name, random_acts):
        """Compute the Concept Activation Vector (CAV) for a concept at a layer.

        This is the core of Visual-TCAV's concept representation. Two approaches
        are supported, controlled by self.cav_method:

        APPROACH 1 — "centroid" (Visual-TCAV original, De Santis et al.):
            1. Extract feature maps for concept and random images at the target layer.
            2. Global Average Pool (GAP) each: [N, C, H, W] -> [N, C].
            3. Compute centroids: mean over N -> [C].
            4. Direction = centroid_concept - centroid_random.
            5. Concept emblems: contraharmonic mean of spatial projections,
               used to normalize the concept map to [0, 1].

            The centroid direction points from "generic" (random) to "concept-specific"
            in activation space. This is geometrically intuitive but assumes the
            concept cluster is roughly spherical and well-separated from random.

        APPROACH 2 — "classifier" (standard TCAV, Kim et al. 2018):
            1. Same GAP pooling as above.
            2. Train a linear SVM to classify concept vs random activations.
            3. Direction = classifier.coef_ (the separating hyperplane normal).
            4. Concept emblems computed the same way as centroid (needed for concept maps).

            The classifier approach finds the *optimal* linear separator, which may
            differ from the centroid direction when the distributions overlap or have
            different variances. It also provides an accuracy metric for the CAV quality.

        WHEN TO USE WHICH:
            - "centroid": fast, deterministic, good for clean concept sets, matches
              the original paper exactly.
            - "classifier": better when concept and random distributions overlap
              significantly, provides accuracy metric, matches standard TCAV literature.
              Requires sklearn.

        Args:
            cache: Whether to use joblib caching.
            concept_name: Name of the concept.
            layer_name: Name of the target layer.
            random_acts: Pre-computed random activations [N, C, H, W].

        Returns:
            ConceptLayer with cav.direction, cav.concept_emblem, and optionally
            cav.accuracy (classifier method only).
        """
        cache_path = os.path.join(
            self.cache_dir,
            f'cav_{concept_name}_{self.model.max_examples}_{self.random_images_folder}_{layer_name}.joblib'
        )
        if cache and os.path.isfile(cache_path):
            concept_layer = load(cache_path)
        else:
            # Get concept activations at this layer
            concept_acts = self.model.activation_generator.get_feature_maps_for_concept(
                concept_name, layer_name
            )

            # Convert to torch tensors if they aren't already
            if not isinstance(concept_acts, torch.Tensor):
                concept_acts = torch.tensor(concept_acts, dtype=torch.float32, device=self.device)
            if not isinstance(random_acts, torch.Tensor):
                random_acts = torch.tensor(random_acts, dtype=torch.float32, device=self.device)

            # Global Average Pooling over spatial dimensions
            # TF format: [N, H, W, C] -> reduce_mean over (1,2) -> [N, C]
            # PyTorch format: [N, C, H, W] -> reduce_mean over (2,3) -> [N, C]
            #
            # Note: The original TF code uses axis=(1,2) because TF uses channels-last.
            # PyTorch uses channels-first, so we reduce over dims (2,3).
            pooled_concept = concept_acts.mean(dim=(2, 3))  # [N, C]
            pooled_random = random_acts.mean(dim=(2, 3))    # [N, C]

            concept_layer = ConceptLayer()

            if self.cav_method == "centroid":
                # CENTROID-BASED CAV (Visual-TCAV original)
                # Direction is simply the vector between cluster centroids.
                # This is equivalent to the TF code:
                #   centroid0 = tf.reduce_mean(pooled_concept, axis=0)
                #   centroid1 = tf.reduce_mean(pooled_random, axis=0)
                #   direction = centroid0 - centroid1
                concept_layer.cav.centroid0 = pooled_concept.mean(dim=0)  # [C]
                concept_layer.cav.centroid1 = pooled_random.mean(dim=0)   # [C]
                concept_layer.cav.direction = concept_layer.cav.centroid0 - concept_layer.cav.centroid1

            elif self.cav_method == "classifier":
                # CLASSIFIER-BASED CAV (standard TCAV approach)
                # Train a linear SVM on pooled activations to find the best
                # separating hyperplane. The normal to this hyperplane is the CAV.
                #
                # This is how Captum's own TCAV computes CAVs internally
                # (see captum.concept._utils.classifier), but we do it ourselves
                # to stay compatible with the rest of the Visual-TCAV pipeline.
                X_concept = pooled_concept.cpu().numpy()
                X_random = pooled_random.cpu().numpy()
                X = np.concatenate([X_concept, X_random], axis=0)
                y = np.concatenate([
                    np.ones(len(X_concept)),
                    np.zeros(len(X_random))
                ])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                clf = LinearSVC(max_iter=10000)
                clf.fit(X_train, y_train)

                # The hyperplane normal IS the CAV direction
                direction = torch.tensor(
                    clf.coef_[0], dtype=torch.float32, device=self.device
                )
                concept_layer.cav.direction = direction

                # Store centroids too (useful for inspection)
                concept_layer.cav.centroid0 = pooled_concept.mean(dim=0)
                concept_layer.cav.centroid1 = pooled_random.mean(dim=0)

                # Store classifier accuracy as a quality metric
                concept_layer.cav.accuracy = accuracy_score(y_test, clf.predict(X_test))

            # CONCEPT EMBLEMS — identical for both methods.
            # Emblems are the 50th percentile of the contraharmonic mean of spatial
            # projections. They define the normalization range for concept maps.
            #
            # The projection of concept_acts onto the CAV direction gives a per-location
            # "concept score". The contraharmonic mean aggregates this spatially,
            # and the median (50th percentile) gives a robust threshold.
            #
            # TF original:
            #   emblems = contraharmonic_mean(
            #       relu(reduce_sum(direction[None,None,None,:] * concept_acts, axis=3)),
            #       axis=(1,2))
            #
            # PyTorch equivalent (channels-first): direction[None,:,None,None] * acts -> sum over C (dim=1)
            proj_concept = F.relu(
                # TODO: torch.mul()
                (concept_layer.cav.direction[None, :, None, None] * concept_acts).sum(dim=1)
            )  # [N, H, W]
            proj_random = F.relu(
                # TODO: torch.mul()
                (concept_layer.cav.direction[None, :, None, None] * random_acts).sum(dim=1)
            )  # [N, H, W]

            # Contraharmonic mean over spatial dims (H, W) -> [N]
            emblems = contraharmonic_mean(proj_concept, dim=(1, 2))
            negative_emblems = contraharmonic_mean(proj_random, dim=(1, 2))

            # 50th percentile (median) as the emblem threshold
            # TF used tfp.stats.percentile; torch.quantile is equivalent.
            concept_layer.cav.concept_emblem = (
                torch.quantile(emblems.float(), 0.5).item(),
                torch.quantile(negative_emblems.float(), 0.5).item(),
            )

            if cache:
                dump(concept_layer, cache_path, compress=3)

        return concept_layer

    def _compute_concept_map(self, feature_maps, direction, concept_emblem):
        """Compute the spatial concept map for a single image's feature maps.

        The concept map shows WHERE in the image a concept is detected, by projecting
        the feature maps onto the CAV direction at each spatial location.

        Args:
            feature_maps: Tensor [C, H, W] — activations at target layer.
            direction: Tensor [C] — the CAV direction vector.
            concept_emblem: tuple (pos_median, neg_median) — normalization thresholds.

        Returns:
            concept_map: Tensor [H, W] in [0, 1].
        """
        # Project feature maps onto CAV direction at each spatial location
        # TF: relu(reduce_sum(direction[None,None,:] * fmaps, axis=2)) on [H,W,C]
        # PyTorch: relu(sum(direction[:,None,None] * fmaps, dim=0)) on [C,H,W]
        concept_map = F.relu(
            (direction[:, None, None] * feature_maps).sum(dim=0)
        )  # [H, W]

        # Normalize using concept emblems
        # The emblem range [neg_median, pos_median] maps to [0, 1].
        # Values outside this range are clipped.
        if concept_emblem[0] > concept_emblem[1]:
            concept_map = torch.clamp(concept_map, min=concept_emblem[1], max=concept_emblem[0])
            concept_map = (concept_map - concept_emblem[1]) / (concept_emblem[0] - concept_emblem[1])
        else:
            # If pos_median <= neg_median, the concept is not meaningfully present.
            # Zero out the map.
            concept_map = concept_map * 0

        return concept_map

    def _compute_masked_attribution(self, attributions, concept_map, feature_maps, direction):
        """Compute the final scalar concept attribution by masking and CAV-weighting.

        This is the final step of the Visual-TCAV pipeline:
        1. Mask the IG attributions by the concept map (spatial filtering).
        2. Normalize the CAV direction (with sign correction for negative activations).
        3. Dot product of normalized CAV with masked attributions -> scalar score.

        Args:
            attributions: Tensor [C, H, W] — processed IG attributions for one class.
            concept_map: Tensor [H, W] — normalized spatial concept map.
            feature_maps: Tensor [C, H, W] — layer activations (for sign correction).
            direction: Tensor [C] — CAV direction vector.

        Returns:
            Scalar attribution value (float tensor).
        """
        # Mask attributions by concept map
        # TF: attributions * concept_map[:,:,None] -> [H,W,C], then sum over (0,1)
        # PyTorch: attributions * concept_map[None,:,:] -> [C,H,W], then sum over (1,2)
        masked_attributions = attributions * concept_map[None, :, :]
        pooled_masked = masked_attributions.sum(dim=(1, 2))  # [C]

        # Normalize CAV direction with sign correction
        # If feature maps have negative values (e.g., after BatchNorm), we need to
        # flip the CAV direction for channels where the concept-weighted features are negative.
        # This handles models without ReLU at the target layer.
        if feature_maps.min() < 0:
            # Sign correction: check if concept-weighted features are negative per channel
            sign = torch.where(
                (feature_maps * concept_map[None, :, :]).sum(dim=(1, 2)) < 0,
                torch.tensor(-1.0, device=feature_maps.device),
                torch.tensor(1.0, device=feature_maps.device),
            )
            pooled_cav_norm = F.relu(direction * sign)
        else:
            pooled_cav_norm = F.relu(direction)

        # Normalize to [0, 1] range
        max_cav = pooled_cav_norm.max()
        if max_cav > 0:
            pooled_cav_norm = pooled_cav_norm / max_cav

        # Final scalar attribution: dot product of normalized CAV and masked attributions
        return torch.dot(pooled_cav_norm, pooled_masked)


#####
# LocalVisualTCAV
#####

class LocalVisualTCAV(VisualTCAV):
    """Local Visual-TCAV: explains a single test image.

    Computes per-concept, per-layer, per-class attributions and spatial concept maps.
    """

    def __init__(
        self,
        test_image_filename, m_steps=50, n_classes=3, target_class=None,
        *args, **kwargs
    ):
        """Initialize local explainer.

        Args:
            test_image_filename: Filename of the test image (relative to test_images_dir).
            m_steps: Number of interpolation steps for integrated gradients.
                Higher = more accurate but slower. 50 is the standard default.
            n_classes: Number of top classes to explain (max 3 for non-binary).
            target_class: If set, explain only this specific class.
        """
        super().__init__(**kwargs)

        self.test_image_filename = test_image_filename
        self.m_steps = m_steps
        self.target_class = target_class

        if self.target_class is not None:
            self.n_classes = 1
            self.target_class_index = self.model.model_wrapper.label_to_id(self.target_class)
        elif not self.model.binary_classification:
            self.n_classes = max(min(n_classes, len(self.model.model_wrapper.labels), 3), 1)
        else:
            self.n_classes = 2

        self.test_images_dir = os.path.join(self.test_images_dir, self.test_image_filename)
        self.resized_imgs_size = self.model.model_wrapper.get_image_shape()[:2]

        self.predictions = []
        self.computations = {}

        # Load and resize the image
        # Note: TF used tf.io.gfile.GFile for cross-platform I/O.
        # In PyTorch we use PIL directly, which handles local files natively.
        # Convert to grayscale or RGB based on model's expected input channels.
        n_channels = self.model.model_wrapper.get_image_shape()[2]
        color_mode = 'L' if n_channels == 1 else 'RGB'
        img = PIL.Image.open(self.test_images_dir).convert(color_mode)
        self.imgs = np.array([img])
        self.resized_imgs = np.array([
            img.resize(self.resized_imgs_size, PIL.Image.BILINEAR)
        ])

    def explain(self, cache_cav=True, cache_random=True, cav_only=False):
        """Run the Visual-TCAV explanation pipeline.

        This is the main orchestration method. For each (layer, concept, class)
        combination, it computes:
            1. CAV direction (centroid or classifier based)
            2. Spatial concept map
            3. Integrated gradients (via Captum)
            4. Masked concept attribution

        Args:
            cache_cav: Cache computed CAVs to disk.
            cache_random: Cache random activations to disk.
            cav_only: If True, only compute CAVs (skip IG and attributions).
        """
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")

        self.computations = {}

        # Prepare the input tensor for Captum's IG
        # Captum needs the original model input (not intermediate features) to compute
        # layer-level IG. We preprocess and store it for _compute_integrated_gradients().
        preprocessed = self.model.preprocessing_function(self.resized_imgs)
        if not isinstance(preprocessed, torch.Tensor):
            preprocessed = torch.tensor(preprocessed, dtype=torch.float32, device=self.device)
        
        # FIX: Ensure it's 4D. If [1, H, W] (grayscale missing channel dim), make it [1, 1, H, W]
        if preprocessed.ndim == 3 and preprocessed.shape[0] == 1:
            preprocessed = preprocessed.unsqueeze(1)

        # Handle channels-last -> channels-first if needed
        if preprocessed.ndim == 4 and preprocessed.shape[-1] in (1, 3):
            preprocessed = preprocessed.permute(0, 3, 1, 2)
            
        self._current_input_tensor = preprocessed.to(self.device)

        for layer_name in tqdm(self.layers, desc="Layers", position=0):
            self.computations[layer_name] = {}

            # Random activations (negative examples for CAV)
            random_acts = self._compute_random_activations(cache_random, layer_name)

            # Get feature maps at this layer for the test image
            # Using Captum's LayerActivation instead of manual hooks
            layer_module = self.model.model_wrapper.get_layer_module(layer_name)
            layer_act = LayerActivation(self.model.model_wrapper.model, layer_module)
            feature_maps = layer_act.attribute(self._current_input_tensor)[0]  # [C, H, W]

            # Compute CAVs for each concept
            for concept_name in self.concepts:
                # CAVs
                concept_layer = self._compute_cavs(cache_cav, concept_name, layer_name, random_acts)

                if not cav_only:
                    # Compute spatial concept map
                    concept_layer.concept_map = self._compute_concept_map(
                        feature_maps, concept_layer.cav.direction, concept_layer.cav.concept_emblem
                    )

                self.computations[layer_name][concept_name] = concept_layer

            if not cav_only:
                # Compute integrated gradients and attributions per class
                attributions = {}
                for n_class in range(self.n_classes):
                    # For non-binary models: compute expected IG magnitude for normalization
                    if not self.model.binary_classification:
                        # Get logits at baseline vs actual feature maps
                        logits = self.model.model_wrapper.get_logits(feature_maps.unsqueeze(0), layer_name)[0]
                        baseline_fmaps = torch.zeros_like(feature_maps)
                        logits_baseline = self.model.model_wrapper.get_logits(baseline_fmaps.unsqueeze(0), layer_name)[0]

                        ig_expected = F.relu(logits - logits_baseline)
                        ig_max = ig_expected.max()
                        if(ig_max > 0):
                            ig_expected_norm =  ig_expected / ig_max  
                        else:
                            ig_expected_norm = ig_expected
                            
                        if self.target_class is not None:
                            ig_expected_class = ig_expected_norm[self.target_class_index]
                        else:
                            ig_expected_class = ig_expected_norm[self.predictions[0][n_class].class_index]

                    # Determine target class index for IG
                    if self.target_class is not None:
                        class_idx = self.target_class_index
                    elif self.model.binary_classification:
                        class_idx = self.predictions[0][0].class_index
                    else:
                        class_idx = self.predictions[0][n_class].class_index

                    # Compute integrated gradients via Captum
                    ig = self._compute_integrated_gradients(feature_maps, layer_name, class_idx)

                    # Compute attributions
                    if self.model.binary_classification:
                        # Binary classification: split IG into positive/negative contributions
                        # representing evidence for/against the predicted class.
                        # TODO: maybe use torch.mult() 
                        binary_attr = ig * feature_maps
                        vl_pos = F.relu(binary_attr).sum()
                        vl_neg = F.relu(-binary_attr).sum()
                        max_vl = max(vl_pos.item(), vl_neg.item())
                        if max_vl > 0:
                            vl_pos = vl_pos / max_vl
                            vl_neg = vl_neg / max_vl

                        if n_class == 0:
                            attr = F.relu(binary_attr)
                            attributions[n_class] = attr / (attr.sum() + 1e-7) * vl_pos
                        else:
                            attr = F.relu(-binary_attr)
                            attributions[n_class] = attr / (attr.sum() + 1e-7) * vl_neg
                    else:
                        # Multi-class: use positive IG * feature_maps, scaled by expected IG
                        attr = F.relu(ig * feature_maps)
                        attributions[n_class] = attr / (attr.sum() + 1e-7) * ig_expected_class

                # Compute final concept attributions by masking with concept maps
                for concept_name in self.concepts:
                    for n_class in range(self.n_classes):
                        self.computations[layer_name][concept_name].attributions[n_class] = \
                            self._compute_masked_attribution(
                                attributions[n_class],
                                self.computations[layer_name][concept_name].concept_map,
                                feature_maps,
                                self.computations[layer_name][concept_name].cav.direction,
                            )

    def plot(self, paper=False):
        """Plot heatmaps and attribution tables for each concept.

        Generates one figure per concept showing:
        - Row 0: Attribution table per class
        - Row 1: Concept heatmap overlaid on the original image
        - Row 2: Example concept images (unless paper=True)
        """
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not len(self.predictions):
            raise Exception("Please let the model predict the classes first")
        if not self.computations:
            raise Exception("Please let the model explain first")

        model_name_esc = self.model.model_name.replace("_", r"\_")

        for concept_name in self.concepts:
            concept_name_esc = concept_name.replace("_", r"\_")

            if not paper:
                fig = plt.figure(figsize=(4 + 5 * (len(self.layers) - 1) - 2 * (len(self.layers) - 1), 7))
                gs = GridSpec(3, len(self.layers) * 3, height_ratios=[2, 5, 2])
                fig.suptitle(
                    f"$\\mathbfit{{{model_name_esc}}}$ architecture\n$\\mathbfit{{{concept_name_esc}}}$ concept",
                    fontsize=10 + 1.5 * len(self.layers)
                )
            else:
                fig = plt.figure(figsize=(4 + 5 * (len(self.layers) - 1) - 2 * (len(self.layers) - 1), 6))
                gs = GridSpec(2, len(self.layers) * 3, height_ratios=[1, 3.5])

            # Example concept images
            if not paper:
                concept_images = self.model.activation_generator.get_images_for_concept(concept_name, False)
                for i in range(min(len(concept_images), len(self.layers) * 3)):
                    fig.add_subplot(gs[2, i])
                    plt.imshow(concept_images[i])
                    plt.tight_layout()
                    plt.axis('off')

            for j, layer_name in enumerate(self.layers):
                layer_description = "" if len(layer_name) > 11 else "layer"
                layer_name_esc = layer_name.replace("_", r"\_")

                concept_layer = self.computations[layer_name][concept_name]

                # Resize concept map to original image size for visualization
                # TF used tf.image.resize; here we use PIL for consistency
                cmap_np = concept_layer.concept_map.cpu().numpy()
                max_value = np.max(cmap_np)

                heatmap = np.array(
                    PIL.Image.fromarray(
                        np.uint8(cmap_np * 255), 'L'
                    ).resize(
                        (self.imgs[0].shape[1], self.imgs[0].shape[0]),
                        PIL.Image.BILINEAR
                    ).filter(
                        PIL.ImageFilter.GaussianBlur(radius=20)
                    )
                ) / 255

                if np.max(heatmap) > 0 and np.max(heatmap) < max_value:
                    heatmap = (heatmap / np.max(heatmap)) * max_value

                # Heatmap subplot
                ax = fig.add_subplot(gs[1, j * 3:(j + 1) * 3])
                plt.imshow(self.imgs[0])
                im = colormap.imshow(heatmap)
                
                if not paper:
                    plt.title(f"\n", fontsize=1)
                
                # Turn off the axes for the image itself
                ax.axis('off') 
                
                # Add the colorbar legend
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Concept Activation (Red = High)', rotation=270, labelpad=15)
                
                plt.tight_layout()

                # Attribution table subplot
                fig.add_subplot(gs[0, j * 3:(j + 1) * 3])
                plt.title(
                    f"$\\mathbfit{{{layer_name_esc}}}$ {layer_description}",
                    fontsize=9 + 1.5 * len(self.layers), y=0.95
                )
                plt.tight_layout()

                rows = []
                for c in range(self.n_classes):
                    attribution = concept_layer.attributions[c]
                    # Convert tensor to float for formatting
                    if isinstance(attribution, torch.Tensor):
                        attribution = attribution.item()

                    if self.target_class is not None:
                        class_name = self.target_class.replace("-", " ")
                    elif self.model.binary_classification and c == 1:
                        class_name = "Not " + self.predictions[0][0].class_name.replace("-", " ")
                    else:
                        class_name = self.predictions[0][c].class_name.replace("-", " ")

                    if len(class_name) > 12:
                        class_name = class_name[:12] + "\u2025"
                    class_name = class_name.replace("_", r"\_").replace(" ", r"\ ")

                    row = [f"$\\mathit{{{class_name}}}$"]
                    if paper:
                        attr_str = f"{attribution:.2g}" if (attribution >= 0.001 or attribution == 0.0) else f"{attribution:.1e}"
                    else:
                        attr_str = f"{attribution:.2g}" if attribution >= 0.001 else f"{attribution:.1e}"
                    attr_str = attr_str.replace("e-0", "e-").replace('-', '{-}')
                    row.append(f"$\\mathbf{{{attr_str}}}$")
                    rows.append(row)

                cols = [f"$\\mathbf{{Class}}$", f"$\\mathbf{{Attrib.}}$"]
                table = plt.table(
                    cellText=rows,
                    rowLabels=[f"" for c in range(self.n_classes)],
                    colLabels=cols,
                    rowColours=["silver"] * 10,
                    colColours=["silver"] * 10,
                    cellLoc='center',
                    rowLoc='center',
                    loc='center', edges='BRTL'
                )
                cellDict = table.get_celld()
                for i in range(0, len(rows) + 1):
                    cellDict[(i, 0)].set_width(.625)
                for i in range(0, len(rows) + 1):
                    cellDict[(i, 1)].set_width(.375)
                for i in range(0, len(cols)):
                    cellDict[(0, i)].set_height(.2)
                    for jj in range(1, self.n_classes + 1):
                        cellDict[(jj, i)].set_height(.2)

                if paper:
                    table.auto_set_font_size(False)
                    table.set_fontsize(4.5 + 1.75 * len(self.layers))
                else:
                    table.set_fontsize(9 + 1.75 * len(self.layers))

                plt.tight_layout()
                plt.axis('off')

            fig.tight_layout()
            plt.show()

    def getCAVs(self, layer_name, concept_name):
        """Get the computed CAV for a specific layer and concept."""
        if not self.computations:
            raise Exception("Please let the model explain first")
        return self.computations[layer_name][concept_name].cav


#####
# GlobalVisualTCAV
#####

class GlobalVisualTCAV(VisualTCAV):
    """Global Visual-TCAV: aggregates concept attributions across multiple test images.

    Computes per-concept mean attribution with confidence intervals.
    """

    def __init__(
        self,
        target_class, test_images_folder, m_steps=50, compute_negative_class=False,
        *args, **kwargs
    ):
        """Initialize global explainer.

        Args:
            target_class: The class to explain (string label).
            test_images_folder: Folder name containing test images (relative to test_images_dir).
            m_steps: Number of interpolation steps for integrated gradients.
            compute_negative_class: If True (binary only), compute attribution for the
                negative class instead of the positive class.
        """
        super().__init__(**kwargs)

        self.m_steps = m_steps
        self.target_class = target_class
        self.compute_negative_class = compute_negative_class
        self.test_images_folder = test_images_folder
        self.test_image_filename = test_images_folder
        self.class_index = self.model.model_wrapper.label_to_id(target_class)

        self.resized_imgs_size = self.model.model_wrapper.get_image_shape()[:2]

        self.predictions = []
        self.stats = {}

    def explain(self, cache_cav=True, cache_random=True):
        """Run global Visual-TCAV explanation across all test images.

        For each (layer, image, concept) triple:
        1. Compute CAVs (once per concept, shared across images).
        2. Compute IG and attributions per image.
        3. Aggregate into Stat objects (mean, std, CI).
        """
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")

        self.stats = {}

        for layer_name in tqdm(self.layers, desc="Layers", position=0):
            self.stats[layer_name] = {}

            # Random activations
            random_acts = self._compute_random_activations(cache_random, layer_name)

            # Get feature maps for ALL test images at this layer
            class_feature_maps = self.computeFeatureMaps(layer_name)

            # Compute CAVs (once per concept, shared across all test images)
            cavs = {}
            attribution_list = {}
            for concept_name in self.concepts:
                concept_layer = self._compute_cavs(cache_cav, concept_name, layer_name, random_acts)
                cavs[concept_name] = concept_layer
                attribution_list[concept_name] = {}

            # For each test image
            for cl, feature_maps in enumerate(tqdm(class_feature_maps, desc="Attributions", position=1)):

                # Prepare input tensor for Captum IG
                # For global explanations, we need the preprocessed input image.
                # We load and preprocess each test image individually.
                self._current_input_tensor = self._get_test_image_tensor(cl)

                if not self.model.binary_classification:
                    # Multi-class: compute expected IG for normalization
                    logits = self.model.model_wrapper.get_logits(
                        feature_maps.unsqueeze(0), layer_name
                    )[0]
                    logits_baseline = self.model.model_wrapper.get_logits(
                        torch.zeros_like(feature_maps).unsqueeze(0), layer_name
                    )[0]

                    ig_expected = F.relu(logits - logits_baseline)
                    ig_max = ig_expected.max()
                    ig_expected_norm = ig_expected / ig_max if ig_max > 0 else ig_expected
                    ig_expected_class = ig_expected_norm[self.class_index]

                # Compute IG
                ig = self._compute_integrated_gradients(feature_maps, layer_name, self.class_index)

                if self.model.binary_classification:
                    binary_attr = ig * feature_maps
                    vl_pos = F.relu(binary_attr).sum()
                    vl_neg = F.relu(-binary_attr).sum()
                    max_vl = max(vl_pos.item(), vl_neg.item())
                    if max_vl > 0:
                        vl_pos = vl_pos / max_vl
                        vl_neg = vl_neg / max_vl

                    if not self.compute_negative_class:
                        attr = F.relu(binary_attr)
                        attributions = attr / (attr.sum() + 1e-7) * vl_pos
                    else:
                        attr = F.relu(-binary_attr)
                        attributions = attr / (attr.sum() + 1e-7) * vl_neg
                else:
                    attr = F.relu(ig * feature_maps)
                    attributions = attr / (attr.sum() + 1e-7) * ig_expected_class

                # Compute concept attribution for each concept
                for concept_name in self.concepts:
                    concept_map = self._compute_concept_map(
                        feature_maps,
                        cavs[concept_name].cav.direction,
                        cavs[concept_name].cav.concept_emblem,
                    )

                    attribution_list[concept_name][cl] = self._compute_masked_attribution(
                        attributions,
                        concept_map,
                        feature_maps,
                        cavs[concept_name].cav.direction,
                    )

            # Aggregate statistics
            for concept_name in self.concepts:
                self.stats[layer_name][concept_name] = Stat(
                    list(attribution_list[concept_name].values())
                )

            del cavs
            del attribution_list

    def _get_test_image_tensor(self, image_index):
        """Load and preprocess a specific test image for Captum IG.

        For global explanations, we need the full model input (not just feature maps)
        so that Captum can compute layer-level IG through the full forward pass.
        """
        # Get the list of test image paths
        test_dir = os.path.join(self.test_images_dir, self.test_images_folder)
        img_paths = sorted([
            os.path.join(test_dir, f) for f in os.listdir(test_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
        ])

        if image_index < len(img_paths):
            n_channels = self.model.model_wrapper.get_image_shape()[2]
            color_mode = 'L' if n_channels == 1 else 'RGB'
            img = PIL.Image.open(img_paths[image_index]).convert(color_mode)
            img = img.resize(self.resized_imgs_size, PIL.Image.BILINEAR)
            img_array = np.array([img])
            preprocessed = self.model.preprocessing_function(img_array)
            if not isinstance(preprocessed, torch.Tensor):
                preprocessed = torch.tensor(preprocessed, dtype=torch.float32, device=self.device)
            
            # FIX: Ensure it's 4D for grayscale images
            if preprocessed.ndim == 3 and preprocessed.shape[0] == 1:
                preprocessed = preprocessed.unsqueeze(1)
                
            # Handle channels-last -> channels-first
            if preprocessed.ndim == 4 and preprocessed.shape[-1] in (1, 3):
                preprocessed = preprocessed.permute(0, 3, 1, 2)
            return preprocessed.to(self.device)
        
        else:
            raise IndexError(f"Image index {image_index} out of range for {len(img_paths)} test images")

    def computeFeatureMaps(self, layer_name):
        """Compute feature maps for all test images at a layer.

        Temporarily redirects the activation generator to the test images directory.
        """
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not layer_name:
            raise Exception("Please provide the function with one layer")

        self.model.activation_generator.concept_images_dir = self.test_images_dir
        class_feature_maps = self.model.activation_generator.get_feature_maps_for_concept(
            self.test_images_folder, layer_name
        )
        self.model.activation_generator.concept_images_dir = self.concept_images_dir

        return class_feature_maps

    def plot(self, colormap_name='viridis', paper=False):
        """Plot bar chart of global concept attributions with error bars."""
        if not self.model:
            raise Exception("Instantiate a Model first")
        if not self.layers or not self.concepts:
            raise Exception("Please add at least one concept and one layer first")
        if not self.stats:
            raise Exception("Please let the model explain first")

        cmap = plt.get_cmap(colormap_name)

        model_name_esc = self.model.model_name.replace("_", r"\_").replace("-", "{-}").replace(" ", r"\text{ }")
        target_class_esc = self.target_class.replace("_", r"\_").replace("-", "{-}").replace(" ", r"\text{ }")
        if self.model.binary_classification and self.compute_negative_class:
            target_class_esc = "Not " + target_class_esc
        concept_names = [
            c.replace("_", r"\_").replace("-", "{-}").replace(" ", r"\text{ }")
            for c in self.concepts
        ]

        fig = plt.figure(figsize=(5 + 1 * (len(self.concepts) - 1), 4))
        gs = GridSpec(1, 1, height_ratios=[1])
        fig.suptitle(
            f"$\\mathbfit{{{model_name_esc}}}$ architecture\n$\\mathbfit{{{target_class_esc}}}$ target class",
            fontsize=12
        )
        fig.add_subplot(gs[0])

        x = np.arange(len(self.concepts)) - 0.5

        for i, layer_name in enumerate(self.layers):
            color = cmap(i / (len(self.layers) - 1)) if len(self.layers) > 1 else cmap(0.5)
            width = 0.1
            pos_x = 0.5 + (i - len(self.layers) / 2) * width + width / 2

            layer_name_esc = layer_name.replace("_", r"\_").replace("-", "{-}").replace(" ", r"\text{ }")

            means = [
                (self.stats[layer_name][cn].begin + self.stats[layer_name][cn].end) / 2
                for cn in self.concepts
            ]
            errs = [
                max(0, (self.stats[layer_name][cn].end - self.stats[layer_name][cn].begin) / 2)
                for cn in self.concepts
            ]

            # Convert tensors to floats for matplotlib
            means = [m.item() if isinstance(m, torch.Tensor) else float(m) for m in means]
            errs = [e.item() if isinstance(e, torch.Tensor) else float(e) for e in errs]

            plt.bar(
                x + pos_x, means, yerr=errs, width=width,
                label=f'$\\mathit{{{layer_name_esc}}}$',
                zorder=2, capsize=3.5, color=color
            )

        plt.ylabel('Attribution (2\u03c3 error)')
        plt.xticks(
            np.arange(len(self.concepts)),
            [f'$\\mathit{{{c}}}$' for c in concept_names]
        )
        plt.grid(linewidth=0.3, zorder=1)
        if paper:
            plt.legend()
        else:
            plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left', borderaxespad=0.0)
        fig.tight_layout()
        plt.ylim(bottom=0, top=max(0.1, plt.ylim()[1]))
        plt.xlim(left=-0.5, right=0.5 + len(self.concepts) - 1)
        plt.show()

    def statsInfo(self):
        """Print a summary table of global attribution statistics using Pandas."""
        if not self.stats:
            raise Exception("Please let the model explain first")

        data = []
        for i, concept_name in enumerate(self.concepts):
            for j, layer_name in enumerate(self.layers):
                stat = self.stats[layer_name][concept_name]
                mean_val = stat.mean.item() if isinstance(stat.mean, torch.Tensor) else stat.mean
                std_val = stat.std if isinstance(stat.std, (int, float)) else stat.std.item()
                begin_val = stat.begin.item() if isinstance(stat.begin, torch.Tensor) else stat.begin
                end_val = stat.end.item() if isinstance(stat.end, torch.Tensor) else stat.end
                
                data.append({
                    "Concept": concept_name if j == 0 else "",
                    "Layer": layer_name,
                    "Attrib. mean": f"{mean_val:.3g} ± {std_val:.3g}",
                    "Attrib. 95.45% CI": f"[{begin_val:.3g}, {end_val:.3g}]"
                })
        
        df = pd.DataFrame(data)
        print(f"\nModel: {self.model.model_name} | Class: {self.target_class} | Folder: {self.test_images_folder}")
        print(df.to_string(index=False))


#####
# Model class
#####

class Model:
    """Wraps a PyTorch model with metadata for Visual-TCAV.

    Unlike the TF version which loads models from .pb/.keras files, the PyTorch
    version accepts either:
    - A path to a saved model/state_dict (loaded via torch.load)
    - A pre-instantiated nn.Module (passed as graph_path_filename)

    The model_wrapper and activation_generator are class references that get
    instantiated by VisualTCAV._bindModel().
    """

    def __init__(
        self,
        model_name,
        graph_path_filename,
        label_path_filename,
        preprocessing_function=lambda x: x / 255.0,
        binary_classification=False,
        max_examples=500,
        image_shape=None,
    ):
        """Initialize model configuration.

        Args:
            model_name: Human-readable model name (used for cache paths, display).
            graph_path_filename: Relative path to model weights file (.pt, .pth),
                or an nn.Module instance directly.
            label_path_filename: Relative path to labels text file (one label per line),
                or a list of label strings directly.
            preprocessing_function: Function applied to images before model input.
                Default divides by 255. For torchvision models, use the appropriate
                transforms (e.g., ImageNet normalization).
            binary_classification: Whether the model is a binary classifier.
            max_examples: Maximum concept/random images to use.
            image_shape: Optional [H, W, C] input shape override. Required for models
                that don't use 224x224 inputs (e.g., [128, 128, 1] for grayscale).
        """
        self.model_name = model_name
        self.max_examples = max_examples
        self.binary_classification = binary_classification
        self.image_shape = image_shape

        self.graph_path_filename = graph_path_filename
        self.label_path_filename = label_path_filename

        self.graph_path_dir = None
        self.label_path_dir = None

        # These are class references, instantiated by _bindModel()
        self.model_wrapper = PytorchModelWrapper
        self.activation_generator = ImageActivationGenerator
        self.preprocessing_function = preprocessing_function

    def getLayerNames(self):
        """Return the names of available intermediate layers."""
        return list(self.model_wrapper.layer_modules.keys())

    def info(self):
        """Print model information table using Pandas."""
        num_classes = len(self.model_wrapper.labels)
        layers = self.getLayerNames()
        
        data = []
        for i, layer_name in enumerate(layers):
            data.append({
                "N. classes": num_classes if i == 0 else "",
                "Layers": layer_name
            })
            
        df = pd.DataFrame(data)
        print(f"\nModel: {self.model_name}")
        print(df.to_string(index=False))


#####
# Data classes
#####

class ConceptLayer:
    """Stores CAV, concept map, and attributions for one (concept, layer) pair."""

    def __init__(self):
        self.attributions = {}
        self.concept_map = None
        self.cav = Cav()


class Cav:
    """Stores a Concept Activation Vector and its metadata.

    Attributes:
        direction: Tensor [C] — the CAV direction in activation space.
        centroid0: Tensor [C] — mean of concept activations (GAP-pooled).
        centroid1: Tensor [C] — mean of random activations (GAP-pooled).
        concept_emblem: tuple (pos_median, neg_median) — normalization thresholds.
        accuracy: float or None — classifier accuracy (only for cav_method="classifier").
    """

    def __init__(self, direction=None, centroid0=None, centroid1=None, concept_emblem=None):
        self.direction = direction
        self.centroid0 = centroid0
        self.centroid1 = centroid1
        self.concept_emblem = concept_emblem
        self.accuracy = None  # Only set when using classifier-based CAVs


class Prediction:
    """A single class prediction with name, index, and confidence."""

    def __init__(self, class_name=None, class_index=None, confidence=None):
        self.class_name = class_name
        self.class_index = class_index
        self.confidence = confidence


class Predictions:
    """Collection of predictions for a test image."""

    def __init__(self, predictions, test_image_filename, model_name):
        self.predictions = predictions
        self.test_image_filename = test_image_filename
        self.model_name = model_name

    def info(self, num_of_classes=3):
        """Print a summary table of top predictions using Pandas."""
        data = []
        for i in range(min(num_of_classes, len(self.predictions[0]))):
            conf = self.predictions[0][i].confidence
            if isinstance(conf, torch.Tensor):
                conf = conf.item()
            
            data.append({
                "Image": self.test_image_filename if i == 0 else "",
                "Class name": self.predictions[0][i].class_name,
                "Confidence": f"{conf:.2f}"
            })
        
        df = pd.DataFrame(data)
        print(f"\nModel: {self.model_name}")
        print(df.to_string(index=False))


class Stat:
    """Statistics for global attribution aggregation.

    Computes mean, std, and 95.45% confidence interval (+/-2 standard errors)
    for a list of attribution values.
    """

    def __init__(self, attributions):
        self.attributions = attributions

        # Convert tensors to numpy for statistics
        vals = np.array([a.item() if isinstance(a, torch.Tensor) else float(a) for a in attributions])

        self.mean = np.mean(vals)
        self.std = np.std(vals)

        self.n = len(vals)
        self.std_err = self.std / np.sqrt(self.n) if self.n > 0 else 0

        # 95.45% CI (+/-2 sigma for normal distribution)
        self.begin = max(0, self.mean - self.std_err * 2)
        self.end = self.mean + self.std_err * 2


#####
# CustomColormap class
#####

class CustomColormap:
    """Custom colormap for concept map visualization.

    The default Visual-TCAV colormap uses black for low values (0-5%) and
    a jet-derived gradient for higher values, making low-activation regions
    transparent when overlaid on the original image.
    """

    def __init__(self, nodes=None, colors=None, min=0, max=1, alpha=0.6):
        if type(nodes) == type(colors):
            if hasattr(nodes, "__len__") and hasattr(colors, "__len__"):
                if len(nodes) != len(colors):
                    raise ValueError('Arrays of different lengths')
            elif nodes is not None or colors is not None:
                raise ValueError('Type not supported')
        else:
            raise ValueError('Attributes of different types')
        if min >= max:
            raise ValueError
        self.nodes = nodes
        self.colors = colors
        self.min = min
        self.max = max
        self.alpha = alpha

    def getLinearSegmentedColormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list("custom", list(zip(self.nodes, self.colors)))

    def imshow(self, heatmap):
        im = plt.imshow(
            heatmap,
            cmap=self.getLinearSegmentedColormap(),
            alpha=self.getAlpha(),
            vmin=self.getMin(),
            vmax=self.getMax()
        )
        plt.clim(self.getMin(), self.getMax())
        return im

    def getMin(self):
        return self.min

    def getMax(self):
        return self.max

    def getAlpha(self):
        return self.alpha


# Default colormap instance (matches TF version)
original_colormap = cm.jet
colormap = CustomColormap(
    nodes=[0.0, 0.05] + [i for i in np.linspace(0.1, 1.0, 100)],
    colors=[(0, 0, 0, 1), (0, 0, 0, 1)] + [original_colormap(i) for i in np.linspace(0.15, 1.0, 100)]
)


#####
# PytorchModelWrapper class
#####

class PytorchModelWrapper():
    """Wraps a PyTorch nn.Module for Visual-TCAV.

    Replaces KerasModelWrapper from the TF version. Key differences:

    TF APPROACH:
        - Model slicing via keras.Model(inputs=..., outputs=layer.output)
        - Separate simulated_layer_model and simulated_logits_model dicts
        - GradientTape for gradient computation

    PYTORCH + CAPTUM APPROACH:
        - LayerActivation for feature map extraction (replaces simulated_layer_model)
        - LayerIntegratedGradients for IG (replaces GradientTape + manual interpolation)
        - get_logits() uses forward hooks instead of model slicing

    The wrapper is model-agnostic: it works with any nn.Module, not just torchvision
    models. Layer selection is done by passing nn.Module references directly to Captum.
    """

    def __init__(self, model_path, labels_path, batch_size, device=None, image_shape=None):
        """Initialize the model wrapper.

        Args:
            model_path: Path to the saved model (.pt, .pth, or full model via torch.save),
                or an nn.Module instance directly.
            labels_path: Path to a text file with one class label per line,
                or a list of label strings.
            batch_size: Batch size for batched forward passes.
            device: torch.device to use.
            image_shape: Optional [H, W, C] override. If None, inferred from first Conv2d.
        """
        self.device = device or DEVICE
        self.batch_size = batch_size
        self.layer_modules = {}  # {layer_name: nn.Module}
        self._image_shape = image_shape  # User override for get_image_shape()

        # Load model
        # Supports three modes:
        #   1. nn.Module instance: used directly (for programmatic use / custom architectures)
        #   2. Full model file: torch.save(model, path) -> torch.load(path)
        #   3. State dict: NOT supported here — user must instantiate architecture first
        if isinstance(model_path, nn.Module):
            self.model = model_path
        elif os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model.to(self.device)
        self.model.eval()

        # Extract layer modules for Captum
        self._get_layer_modules()

        # Load labels
        if isinstance(labels_path, list):
            # Allow passing labels directly as a list
            self.labels = labels_path
        elif os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = f.read().splitlines()
        else:
            raise FileNotFoundError(f"Labels not found at {labels_path}")

    def id_to_label(self, idx):
        """Get the class label from its index."""
        return self.labels[idx]

    def label_to_id(self, label):
        """Get the class index from its label."""
        return self.labels.index(label)

    def get_predictions(self, imgs):
        """Run forward pass and return predictions.

        Args:
            imgs: numpy array or tensor of preprocessed images [N, H, W, C] or [N, C, H, W].

        Returns:
            numpy array of shape [N, num_classes] with prediction scores.
        """
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0) # Becomes [1, H, W, C]

        # Handle channels-last input (legacy compatibility with TF-style preprocessing)
        if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
            imgs = imgs.permute(0, 3, 1, 2)

        imgs = imgs.to(self.device)

        with torch.no_grad():
            predictions = self.model(imgs)

        return predictions.cpu().numpy()

    def get_feature_maps(self, imgs, layer_name):
        """Extract feature maps at a specific layer using Captum's LayerActivation.

        This replaces the TF pattern of:
            simulated_model = keras.Model(inputs=model.input, outputs=layer.output)
            feature_maps = simulated_model(images)

        Captum's LayerActivation handles hook registration internally, so we don't
        need to manage forward hooks ourselves.

        ALTERNATIVE (without Captum):
            You could manually register forward hooks:
                activations = {}
                def hook_fn(module, input, output):
                    activations['value'] = output
                handle = layer_module.register_forward_hook(hook_fn)
                model(imgs)
                handle.remove()
                return activations['value']
            But this requires careful hook lifecycle management (especially with batching
            and error handling). Captum does this correctly and efficiently.

        Args:
            imgs: Preprocessed images [N, H, W, C] or [N, C, H, W].
            layer_name: Name of the target layer.

        Returns:
            Feature maps tensor [N, C, H, W].
        """
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)

        # Handle channels-last input
        if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
            imgs = imgs.permute(0, 3, 1, 2)

        imgs = imgs.to(self.device)

        layer_module = self.get_layer_module(layer_name)
        layer_act = LayerActivation(self.model, layer_module)

        # Batch processing to manage memory
        all_fmaps = []
        for i in range(0, len(imgs), self.batch_size):
            batch = imgs[i:i + self.batch_size]
            with torch.no_grad():
                fmaps = layer_act.attribute(batch)
            all_fmaps.append(fmaps)

        return torch.cat(all_fmaps, dim=0)

    def get_logits(self, feature_maps, layer_name):
        """Compute logits from intermediate feature maps.

        This replaces the TF pattern of building a sub-model from an intermediate
        layer to the output:
            keras.Model(inputs=layer.output, outputs=model.output)

        In PyTorch, we use a forward hook to inject the feature maps at the target
        layer and run the rest of the network.

        Args:
            feature_maps: Tensor [N, C, H, W] — activations at the target layer.
            layer_name: Name of the layer these feature maps came from.

        Returns:
            Logits tensor [N, num_classes].
        """
        layer_module = self.get_layer_module(layer_name)

        # We use a hook to replace the layer's output with our feature maps,
        # then run a forward pass to get the logits.
        # This is equivalent to TF's model slicing but done via hooks.
        def replace_hook(module, input, output):
            return feature_maps

        handle = layer_module.register_forward_hook(replace_hook)

        # We need a dummy input with the right shape to trigger the forward pass
        input_shape = self.get_image_shape()
        dummy = torch.zeros(
            feature_maps.shape[0], input_shape[2], input_shape[0], input_shape[1],
            device=self.device
        )

        with torch.no_grad():
            logits = self.model(dummy)

        handle.remove()
        return logits

    def get_layer_module(self, layer_name):
        """Get the nn.Module for a named layer.

        Args:
            layer_name: Dot-separated module path (e.g., 'layer4.2.conv3').

        Returns:
            The nn.Module at that path.
        """
        if layer_name in self.layer_modules:
            return self.layer_modules[layer_name]

        # Traverse the module tree by splitting on '.'
        module = self.model
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def get_image_shape(self):
        """Get the expected input image shape [H, W, C].

        Returns the user-provided image_shape if set, otherwise infers from the
        first Conv2d layer's input channels with 224x224 default spatial size.
        Unlike TF's model.input_shape, PyTorch models don't store their expected
        input shape explicitly, so an explicit override via image_shape is
        recommended for non-224x224 models.
        """
        if self._image_shape is not None:
            return list(self._image_shape)
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                c = module.in_channels
                return [224, 224, c]
        # Fallback for models without Conv2d (e.g., fully-connected)
        return [224, 224, 3]

    def _get_layer_modules(self):
        """Discover and store intermediate layer modules for Captum.

        This replaces the TF _get_layer_tensors() method. Instead of extracting
        layer.output tensors, we store references to nn.Module objects that Captum's
        LayerActivation / LayerIntegratedGradients can target directly.

        The layer selection logic mirrors the TF version for supported architectures:
        - ResNet: bottleneck/basicblock outputs
        - VGG: conv layers
        - Inception: mixed layers
        - ConvNeXt: residual connection layers
        - Default: all non-trivial layers (Conv2d, Linear, Sequential) up to depth 2
        """
        self.layer_modules = {}
        model_name = self.model.__class__.__name__.lower()

        for name, module in self.model.named_modules():
            if name == '':
                continue

            # Architecture-specific layer selection
            # Mirrors the TF version's _get_layer_tensors() logic
            if 'resnet' in model_name:
                # ResNet: select bottleneck/basicblock outputs (layer1.0, layer1.1, etc.)
                # This matches TF's conv4/conv5 block selection
                if any(f'layer{i}' in name for i in range(1, 5)):
                    parts = name.split('.')
                    if len(parts) == 2 and parts[1].isdigit():
                        self.layer_modules[name] = module

            elif 'vgg' in model_name:
                # VGG: select all Conv2d layers within the features sequential
                if isinstance(module, nn.Conv2d) and 'features' in name:
                    self.layer_modules[name] = module

            elif 'inception' in model_name:
                # Inception: select mixed layers
                if 'mixed' in name.lower() or 'Mix' in name:
                    self.layer_modules[name] = module

            elif 'convnext' in model_name:
                # ConvNeXt: select residual blocks
                if isinstance(module, nn.Module) and name.count('.') == 2:
                    self.layer_modules[name] = module

            else:
                # Default: include "meaningful" layers at reasonable depth
                # This catches custom architectures like FlexibleTNet
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.Sequential)):
                    if name.count('.') <= 2:
                        self.layer_modules[name] = module


#####
# ImageActivationGenerator class
#####

class ImageActivationGenerator():
    """Generates and caches feature map activations for concept images.

    This class loads concept images from directories, preprocesses them,
    and extracts feature maps at target layers using the model wrapper.

    The image loading uses multiprocessing (same as TF version) for parallel I/O.
    The main difference from TF is:
    - PIL.Image.open() instead of tf.io.gfile.GFile for file I/O
    - Returns numpy arrays (converted to tensors by the model wrapper)

    ALTERNATIVE APPROACH (Captum Concept API):
        For new projects, you could replace this entire class with Captum's
        concept data management:

            from captum.concept import Concept
            from captum.concept._utils.data_iterator import (
                CustomIterableDataset, dataset_to_dataloader
            )

            def assemble_concept(name, cid, concepts_path, transform):
                dataset = CustomIterableDataset(
                    lambda f: transform(Image.open(f)).float(),
                    str(concepts_path / name) + "/"
                )
                return Concept(
                    id=cid, name=name,
                    data_iter=dataset_to_dataloader(dataset)
                )

        This is demonstrated in visual_tcav_demo.ipynb. We keep the custom pipeline
        here for backward compatibility with the TF version's directory structure,
        caching strategy, and preprocessing conventions.
    """

    def __init__(
        self,
        model_wrapper,
        concept_images_dir,
        cache_dir,
        preprocessing_function=None,
        max_examples=500,
    ):
        self.model_wrapper = model_wrapper
        self.concept_images_dir = concept_images_dir
        self.cache_dir = cache_dir
        self.max_examples = max_examples
        self.preprocessing_function = preprocessing_function

    def get_feature_maps_for_concept(self, concept, layer):
        """Load concept images and extract feature maps at the given layer."""
        images = self.get_images_for_concept(concept)
        if len(images) == 0:
            raise Exception("Please provide example images for each concept")
        feature_maps = self.model_wrapper.get_feature_maps(images, layer)
        return feature_maps

    def get_feature_maps_for_layers_and_concepts(self, layer_names, concepts, cache=True):
        """Compute or load cached feature maps for all (concept, layer) pairs."""
        feature_maps = {}

        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        for concept in concepts:
            if concept not in feature_maps:
                feature_maps[concept] = {}
            for layer_name in layer_names:
                fmaps_path = os.path.join(
                    self.cache_dir,
                    f'f_maps_{concept}_{layer_name}.joblib'
                ) if self.cache_dir else None

                if fmaps_path and os.path.exists(fmaps_path) and cache:
                    feature_maps[concept][layer_name] = load(fmaps_path)
                else:
                    feature_maps[concept][layer_name] = self.get_feature_maps_for_concept(concept, layer_name)
                    if fmaps_path and cache:
                        os.makedirs(os.path.dirname(fmaps_path), exist_ok=True)
                        dump(feature_maps[concept][layer_name], fmaps_path, compress=3)

        return feature_maps

    def get_images_for_concept(self, concept, preprocess=True):
        """Load images from a concept's directory.

        Args:
            concept: Name of the concept (subdirectory of concept_images_dir).
            preprocess: Whether to apply the preprocessing function.

        Returns:
            numpy array of shape [N, H, W, C].
        """
        concept_dir = os.path.join(self.concept_images_dir, concept)

        def is_image(filename):
            return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))

        img_paths = [
            os.path.join(concept_dir, d)
            for d in sorted(os.listdir(concept_dir))
            if is_image(d)
        ]

        imgs = self._load_images_from_files(
            img_paths,
            self.max_examples,
            shape=self.model_wrapper.get_image_shape()[:2],
            preprocess=preprocess
        )
        return imgs

    def _load_images_from_files(self, filenames, max_imgs=500, shape=(224, 224), preprocess=True):
        """Load multiple images in parallel using multiprocessing."""
        pool = multiprocessing.Pool(50)
        imgs = pool.map(
            lambda filename: self._load_image_from_file(filename, shape, preprocess=preprocess),
            filenames[:max_imgs]
        )
        pool.close()
        imgs = [img for img in imgs if img is not None]
        return np.array(imgs)

    def _load_image_from_file(self, filename, shape, preprocess=True):
        """Load and preprocess a single image file.

        Args:
            filename: Path to the image file.
            shape: Target (H, W) for resizing.
            preprocess: Whether to apply preprocessing_function.

        Returns:
            numpy array [H, W, C] or [H, W] (grayscale) or None on error.
        """
        try:
            # Determine channel mode from model's expected input shape
            n_channels = self.model_wrapper.get_image_shape()[2]
            if n_channels == 1:
                img = np.array(
                    PIL.Image.open(filename).convert('L').resize(shape, PIL.Image.BILINEAR),
                )
                # Add channel dimension: [H, W] -> [H, W, 1]
                img = img[:, :, np.newaxis]
            else:
                img = np.array(
                    PIL.Image.open(filename).convert('RGB').resize(shape, PIL.Image.BILINEAR),
                )
        except Exception:
            return None

        if self.preprocessing_function is not None and preprocess:
            img = self.preprocessing_function(img)

        # Validate shape: must be 3D with expected channel count
        if len(img.shape) == 3 and img.shape[2] == n_channels:
            return img
        # Also accept preprocessed tensors that dropped the channel dim
        elif len(img.shape) == 2 and n_channels == 1:
            return img[:, :, np.newaxis]
        return None
        return img
