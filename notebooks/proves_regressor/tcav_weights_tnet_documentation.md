## Documentation of `tcav_weights_tnet.ipynb`

---

## 1. Overview and Motivation

This notebook implements a complete pipeline for **concept-based interpretability** of a custom convolutional neural network (TNet) using **Testing with Concept Activation Vectors (TCAV)** (Kim et al., 2018). The central objective is to extract, analyse, and spatially visualise the internal representations that TNet has learned for geometric shape concepts (circles, squares, crosses) from the AIXI synthetic dataset.

Unlike standard feature-attribution methods (e.g., saliency maps, Grad-CAM) that operate at the pixel level, TCAV operates in the **activation space** of internal network layers, testing whether user-defined high-level concepts are meaningfully encoded in those representations. This notebook extends the standard TCAV pipeline by:

1. Explicitly extracting the **linear classifier weights** (CAV vectors) from each concept-layer pair.
2. Projecting those weights back onto spatial activation maps to produce **concept-localised heatmaps**.
3. Diagnosing a known failure mode of TCAV under **sigmoid saturation**.

---

## 2. Target Model: TNet Architecture

**TNet** is a custom 5-block convolutional neural network designed for binary classification on 128x128 single-channel (grayscale) images. The model is defined in `model/model_sq.py`.

### 2.1 Architecture Summary

| Block | Layers | Output Shape |
|-------|--------|-------------|
| Conv Block 1 | Conv2d(1, 25, 3x3, same) -> ReLU -> MaxPool2d(2x2) | [25, 64, 64] |
| Conv Block 2 | Conv2d(25, 35, 3x3, same) -> ReLU -> MaxPool2d(2x2) | [35, 32, 32] |
| Conv Block 3 | Conv2d(35, 50, 3x3, same) -> ReLU -> BatchNorm2d -> MaxPool2d(2x2) | [50, 16, 16] |
| Conv Block 4 | Conv2d(50, 75, 3x3, same) -> ReLU -> BatchNorm2d -> MaxPool2d(2x2) | [75, 8, 8] |
| Conv Block 5 | Conv2d(75, 125, 3x3, same) -> ReLU -> BatchNorm2d -> MaxPool2d(2x2) | [125, 4, 4] |
| Flatten | - | [2000] |
| FC Block 1 | Linear(2000, 500) -> Dropout(0.2) -> ReLU | [500] |
| FC Block 2 | Linear(500, 250) -> Dropout(0.2) -> ReLU | [250] |
| FC Block 3 | Linear(250, 50) -> Dropout(0.2) -> ReLU | [50] |
| Output | Linear(50, 1) -> Sigmoid | [1] |

### 2.2 Task Definition

The model performs **binary classification** with sigmoid output:
- **Output ~ 1.0**: The input image "has circles" (circle-present class).
- **Output ~ 0.0**: The input image is "square-dominant with no circles" (label = 0 iff `circles == 0 AND squares > 0 AND squares >= crosses`).

The pretrained weights are loaded from `weights/only_sq.pt`.

---

## 3. Notebook Sections in Detail

### 3.1 Environment Setup and Imports (Cells 0-1)

**Purpose:** Establish the working directory and import all required libraries.

**Key dependencies:**
- **PyTorch**: Model inference and tensor operations.
- **Captum** (`captum.concept`): The TCAV implementation, including `TCAV`, `Concept`, `CustomIterableDataset`, and `dataset_to_dataloader`.
- **NumPy / Matplotlib / PIL**: Numerical computation, visualisation, and image I/O.

The working directory is explicitly set to the project root to ensure consistent relative paths for data and weight files.

---

### 3.2 Model Loading and Initialisation (Cell 2)

**Purpose:** Load the pretrained TNet model and prepare it for inference.

**Method:**
1. The weight dictionary is loaded from disk with `torch.load(..., weights_only=True)` for security.
2. The input channel count and number of output classes are inferred directly from the weight tensor shapes (`conv1.weight.shape[1]` and `fc4.weight.shape[0]`), ensuring the model architecture matches the checkpoint.
3. The model is set to evaluation mode (`model.eval()`), which disables dropout and sets batch normalisation to use running statistics rather than batch statistics.

**Result:** A TNet with `in_channels=1` (grayscale) and `classes=1` (binary sigmoid output), running on CPU.

---

### 3.3 Image Transforms and Concept Data Loader (Cell 3)

**Purpose:** Define the preprocessing pipeline and a reusable factory function for creating Captum `Concept` objects.

**Transform pipeline:**
1. `Grayscale()` -- Ensures all images are single-channel.
2. `Resize((128, 128))` -- Matches TNet's expected input resolution.
3. `ToTensor()` -- Converts PIL images to float32 tensors normalised to [0, 1].

**`assemble_concept(name, id, concepts_path)`:** A factory function that:
1. Constructs the directory path for a named concept (e.g., `data/concepts/circle_full/`).
2. Creates a `CustomIterableDataset` that lazily loads and transforms images from that directory.
3. Wraps it in a `DataLoader` via `dataset_to_dataloader`.
4. Returns a Captum `Concept` object with a unique integer ID, a human-readable name, and the data iterator.

This abstraction allows uniform treatment of both shape concepts and random control pools.

---

### 3.4 Concept Assembly (Cell 4)

**Purpose:** Instantiate the concept datasets for the three geometric shape classes and four random control pools.

**Shape concepts** (experimental):

| Concept | ID | Description |
|---------|----|-------------|
| `circle_full` | 0 | Images containing circles |
| `cross_full` | 1 | Images containing crosses |
| `square_full` | 2 | Images containing squares |

**Random pools** (control / negative class for the linear classifier):

| Pool | ID | Purpose |
|------|----|---------|
| `random_pool` | 100 | Primary random baseline |
| `random_pool_2` | 101 | Additional random pool (available but not used in this notebook) |
| `random_pool_3` | 102 | Additional random pool |
| `random_pool_4` | 103 | Additional random pool |

In TCAV, the random pools serve as the **negative class** when training the linear classifier (CAV). Their purpose is to provide a statistically neutral baseline: if a concept's CAV direction is no more informative than a direction separating random images, the concept is not meaningfully encoded at that layer.

---

### 3.5 Layer Selection and Experimental Set Definition (Cell 5)

**Purpose:** Define which internal layers to probe and which concept vs. random pairings to evaluate.

**Probed layers:** `['conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']`

These span the mid-to-late convolutional blocks (where spatial structure is still preserved) through the fully connected layers (where spatial information has been collapsed). Early layers (`conv1`, `conv2`) are excluded because they tend to encode low-level features (edges, textures) that are less concept-specific.

**Experimental sets** (3 pairs):
```
circle_full  vs. random_pool
cross_full   vs. random_pool
square_full  vs. random_pool
```

Each set defines a binary classification task: "Can a linear classifier, trained on the activations at layer L, distinguish concept images from random images?" The resulting classifier's weight vector defines the **Concept Activation Vector (CAV)** -- the direction in activation space that maximally separates the concept from randomness.

---

### 3.6 CAV Computation (Cell 6)

**Purpose:** Train the linear classifiers and extract the Concept Activation Vectors.

**Method:**
1. A `TCAV` object is instantiated with the model and target layers.
2. `mytcav.compute_cavs(experimental_sets)` performs the following for each (concept, random, layer) triple:
   - Passes all concept images and all random images through the model.
   - Collects the intermediate activations at the specified layer.
   - **Flattens** the activations to 1D vectors (for convolutional layers, this flattens `[C, H, W]` to `[C*H*W]`).
   - Trains a **linear classifier** (logistic regression via Captum's `DefaultClassifier`) to distinguish concept activations from random activations.
   - Stores the trained classifier's weight matrix and accuracy statistics.

**Result structure:** A nested dictionary keyed by `"concept_id-random_id"` (e.g., `"0-100"` for circle_full vs. random_pool), then by layer name, containing `CAV` objects with `.stats['weights']` and `.stats['accs']`.

**Note:** Captum's `DefaultClassifier` stores all training data in memory, which triggers a warning. For larger-scale experiments, a custom `Classifier` subclass with batched training would be advisable.

---

### 3.7 CAV Weight Extraction and Analysis (Cell 7)

**Purpose:** Extract the raw weight vectors from each CAV and compute summary statistics.

**Method:**
For each (concept, layer) pair, the weight matrix of the trained linear classifier has shape `[2, D]` where `D` is the flattened activation dimensionality:
- **Row 0:** Weights for the **concept class** (positive direction).
- **Row 1:** Weights for the **random class** (negative direction).

Only Row 0 is extracted, as it defines the CAV direction -- the direction in activation space along which the concept is maximally activated relative to randomness.

**Key observations from the output:**

| Layer | Dimensionality | Interpretation |
|-------|---------------|----------------|
| `conv3` | 51,200 (50 x 32 x 32) | High-dimensional spatial features; small weight magnitudes (~0.008) indicate diffuse, distributed encoding. |
| `conv4` | 19,200 (75 x 16 x 16) | Intermediate spatial/channel encoding; slightly larger weights (~0.02). |
| `conv5` | 8,000 (125 x 8 x 8) | Late convolutional features; weights increase (~0.2), suggesting more concentrated concept encoding. |
| `fc1` | 500 | First fully connected layer; large weights (~1.0) indicate strong concept-specific neurons. |
| `fc2` | 250 | Highest weight magnitudes (~2.3 for square_full), indicating the most concept-discriminative layer. |
| `fc3` | 50 | Pre-output layer; mixed results (high for circle/square, near-zero for cross). |

The **monotonic increase in max absolute weight** from convolutional to fully connected layers reflects the network's progressive abstraction: early layers encode distributed spatial patterns, while later layers compress information into compact, concept-aligned representations.

---

### 3.8 Top-N CAV Weight Visualisation (Cell 8)

**Purpose:** For each concept, visualise the N=20 largest-magnitude CAV weights at each layer to identify the most concept-discriminative features.

**Method:**
1. For each (concept, layer) pair, sort the weight vector by absolute value in descending order.
2. Select the top 20 weights and their corresponding feature indices.
3. Plot as a bar chart with **blue bars** for positive weights (features that support concept presence) and **red bars** for negative weights (features that support concept absence).

**Interpretation:**
- The feature indices on the x-axis correspond to specific dimensions in the flattened activation vector. For convolutional layers, these can be mapped back to (channel, spatial_row, spatial_col) positions.
- A small number of dominant weights indicates that the concept is encoded by a sparse set of features; a more uniform distribution suggests distributed encoding.
- The sign of the weight indicates whether higher activation of that feature correlates with concept presence (positive) or absence (negative).

---

### 3.9 Summary Heatmap: Mean and Max |Weight| per Concept x Layer (Cell 9)

**Purpose:** Provide a compact overview of how strongly each concept is encoded at each layer.

**Metrics computed:**
1. **Mean |weight|:** The average absolute CAV weight across all dimensions at a given layer. Reflects the overall strength of concept encoding -- higher values indicate that the concept direction is well-separated from random on average.
2. **Max |weight|:** The single largest absolute weight. Indicates whether any individual feature is highly discriminative for the concept.

**Visualisation:** Two side-by-side heatmaps (concepts on y-axis, layers on x-axis) using the `YlOrRd` (Yellow-Orange-Red) colour map, with numeric annotations in each cell.

**Interpretation:**
- Early convolutional layers (`conv3`, `conv4`) show small mean values across all concepts, consistent with distributed spatial encoding.
- Fully connected layers (`fc1`, `fc2`, `fc3`) show markedly higher values, with `fc2` being the peak for most concepts. This suggests `fc2` is the layer where geometric concept information is most compactly represented.

---

### 3.10 Validation Image Loading and Model Classification (Cell 10)

**Purpose:** Load a set of validation images, classify them with the trained model, and partition them into two groups for differential TCAV analysis.

**Method:**
1. Load the first 200 PNG images from `data/aixi_shape/val/`.
2. Apply the same grayscale + resize + tensor transform.
3. Run a batch forward pass through the model (with `torch.no_grad()` for efficiency).
4. Partition images by model prediction:
   - **`circle_mask`** (pred > 0.5): Images the model classifies as "has circles" (138 images).
   - **`sq_dominant_mask`** (pred <= 0.5): Images the model classifies as "square-dominant, no circles" (62 images).
5. Sample N=20 images from each group for subsequent TCAV scoring.

**Rationale:** Separating images by model prediction allows testing whether TCAV scores differ between the two decision regimes, revealing which concepts the model relies on for each class.

---

### 3.11 TCAV Interpretation Scores (Cell 11)

**Purpose:** Compute TCAV scores for each concept on both image groups to quantify concept influence on model predictions.

**Method:**
The `mytcav.interpret()` method computes, for each (concept, layer) pair:

1. **Directional derivative:** For each input image, compute the gradient of the model's output with respect to the activations at layer L, then project this gradient onto the CAV direction. This measures how much moving in the concept direction at layer L would change the model's prediction.
2. **Sign count:** Count how many input images have a **positive** directional derivative (concept direction increases prediction) vs. negative.
3. **TCAV rate:** The fraction of images with positive directional derivative. A rate significantly above 0.5 indicates that the concept positively influences the model's prediction for that class.

**Critical finding -- Sigmoid saturation problem:**

For **circle images** (pred ~ 1.0), all TCAV scores are exactly **zero** across all concepts and layers. This is a consequence of the sigmoid activation function at the output:

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))$$

When the model is highly confident (pred ~ 1.0), the pre-sigmoid logit $x$ is very large, causing $\sigma'(x) \to 0$. Since TCAV's default attribution method (`LayerGradientXActivation`) relies on the gradient flowing back from the output, the vanishing sigmoid derivative nullifies all TCAV scores.

For **square-dominant images** (pred ~ 0.0 to 0.5), the sigmoid is in its transition region where gradients flow, and meaningful TCAV scores emerge:

| Concept | conv3 | conv4 | conv5 | fc1 | fc2 | fc3 |
|---------|-------|-------|-------|-----|-----|-----|
| circle_full | 0.85 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| cross_full | 0.60 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| square_full | 0.75 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |

The high rates in convolutional layers (`conv3-conv5`) and zero rates in fully connected layers suggest that concept sensitivity, as measured by TCAV's gradient-based method, is localised in the spatial feature extraction stages of the network.

---

### 3.12 TCAV Score and CAV Norm Visualisation (Cell 13)

Modifications: Use Captum's TCAV for all of the TCAV plots and usage, use the API for this.

**Purpose:** Side-by-side comparison of TCAV rates and CAV weight norms to contrast gradient-based concept sensitivity with classifier-based concept separability.

**Plot 1 -- TCAV rates:** Bar chart of positive TCAV rates per concept per layer (square-dominant images only). A dashed line at 0.5 marks the chance level. Rates above this line indicate that the concept direction positively influences the model's output.

**Plot 2 -- CAV L2 norms:** $\|w\|_2$ for each concept's CAV at each layer. Higher norms indicate that the linear classifier needed larger weights to separate concept from random activations, which can reflect either:
- **Harder separation** (activations overlap, requiring larger decision boundaries), or
- **Higher-dimensional encoding** (more features contribute to the concept).

**Key insight:** The CAV norms are high in convolutional layers (high-dimensional spaces) and decrease in FC layers, while TCAV rates are high only in conv layers. This divergence highlights that TCAV rates and CAV norms measure fundamentally different properties: gradient-based sensitivity vs. linear separability.

---

### 3.13 Feature Map Visualisation via Forward Hooks (Cells 15-17)

Modifications: Why use Maxpooling and not Global Avg Pooling

**Purpose:** Capture and visualise the intermediate spatial activation maps (feature maps) produced by TNet when processing concept images.

**Method:**
1. **Forward hooks** are registered on all 5 `MaxPool2d` layers using PyTorch's `register_forward_hook` API. Each hook stores the output tensor in a dictionary keyed by layer name.
2. A single representative image from each concept directory (`circle_full`, `cross_full`, `square_full`) is passed through the model.
3. The captured activations are visualised as **mean activation heatmaps** -- the average activation across all channels at each spatial position.

**Activation shapes captured:**

| Hook Layer | Channels | Spatial Dims | Description |
|------------|----------|-------------|-------------|
| maxpool1 | 25 | 64 x 64 | Low-level edge and texture features |
| maxpool2 | 35 | 32 x 32 | Mid-level local patterns |
| maxpool3 | 50 | 16 x 16 | Higher-order shape fragments |
| maxpool4 | 75 | 8 x 8 | Abstract shape representations |
| maxpool5 | 125 | 4 x 4 | Highly compressed global features |

**Visualisation (Cell 17):** A grid with concepts on rows and layers (maxpool3-5) on columns, showing: (a) the original concept image, and (b) heatmaps of mean channel activation, revealing which spatial regions are most strongly activated by each concept.

---

### 3.14 Top CAV-Weighted Channel Activations (Cell 18)

**Purpose:** Identify and visualise the specific activation channels that are most important for each concept, as determined by the CAV weights.

**Method:**
1. For each (concept, convolutional layer) pair, the flattened CAV weight vector is reshaped to `[C, H*W]` where `C` is the number of channels.
2. **Per-channel importance** is computed as the mean absolute weight across all spatial positions for that channel: $\text{importance}(c) = \frac{1}{|S|} \sum_{s \in S} |w_{c,s}|$
3. The top K=5 most important channels are selected.
4. For each concept, a grid shows: the mean activation (column 0) followed by the individual activation maps of the top-5 most important channels.

**Interpretation:** This analysis bridges the gap between the abstract CAV vector and the spatially grounded feature maps. It answers the question: "Which specific learned filters are most responsible for encoding this concept?"

---

### 3.15 CAV-Projected Concept Heatmaps on Validation Images (Cells 19-21)

**Purpose:** Generate spatial heatmaps showing **where in an input image** a specific concept's features are being detected, using the CAV weights as a projection operator.

**Mathematical formulation:**

For a convolutional layer with $C$ channels and spatial dimensions $H \times W$, the CAV weight vector is reshaped to $\mathbf{W} \in \mathbb{R}^{C \times (H \cdot W)}$. The per-channel **signed importance** is:

$$\text{importance}(c) = \frac{1}{|S|} \sum_{s} W_{c,s}$$

The concept heatmap for a given input image is then:

$$\text{heatmap}(h, w) = \sum_{c=1}^{C} \text{importance}(c) \cdot A_{c,h,w}$$

where $A_{c,h,w}$ is the activation of channel $c$ at spatial position $(h, w)$ after the corresponding MaxPool layer.

The heatmap is bilinearly upsampled to the original image resolution (128x128) and overlaid with a `jet` colour map on the grayscale input.

**Three visualisation stages:**

1. **Cell 20 -- Square-dominant images:** Heatmaps are computed for layers where the TCAV rate was high (conv3, conv4, conv5), providing a spatially grounded view of where concept features are detected in images that the model classifies as non-circle.

2. **Cell 21 -- Circle images (TCAV-zero regime):** Despite TCAV scores being zero (due to sigmoid saturation), the CAV projection heatmaps remain informative because they depend on `activations * CAV_weights`, not on output gradients. This demonstrates that CAV spatial projections are a **gradient-free** concept localisation method that circumvents the sigmoid saturation problem.

**Key insight:** This is one of the most valuable contributions of the notebook -- demonstrating that even when TCAV's gradient-based scoring fails (saturated outputs), the learned CAV directions can still be used for concept localisation through direct activation projection.

---

### 3.16 Single Multi-Shape Image Analysis (Cells 22-23)

**Purpose:** Provide a detailed case study of activation patterns for a single validation image containing multiple geometric shapes.

**Method:**
1. The validation CSV (`data/aixi_shape/val/dades.csv`) is read to find an image with known shape counts.
2. Image index 11 is selected: **0 circles, 2 squares, 1 cross** -- a square-dominant, no-circle image (expected label = 0.0).
3. A forward pass captures all 5 maxpool activations via the registered hooks.
4. All 5 mean activation heatmaps are upsampled to 128x128 and overlaid on the original image.

**Visualisation:** A 2x3 grid showing the original image alongside 5 progressively coarser activation heatmaps (maxpool1 through maxpool5), illustrating how the network's spatial attention evolves from fine-grained edge detection to coarse global feature encoding.

---

## 4. Summary of Key Findings

1. **CAV weight magnitude increases with layer depth:** Convolutional layers encode concepts in high-dimensional, distributed representations with small individual weights. Fully connected layers compress this into sparse, high-magnitude representations, with `fc2` showing the strongest concept discriminability.

2. **TCAV scores fail under sigmoid saturation:** When the model's output sigmoid is saturated (pred ~ 0 or ~ 1), the vanishing gradient nullifies all TCAV scores. This is a fundamental limitation of gradient-based TCAV for models with sigmoid outputs.

3. **CAV spatial projections are gradient-free:** By projecting CAV weights directly onto activation maps (without computing output gradients), concept localisation remains functional even in the sigmoid-saturated regime. This provides a complementary and more robust method for concept-based spatial interpretability.

4. **Concept encoding is layer-dependent:** TCAV rates are high in mid-to-late convolutional layers (conv3-conv5) but drop to zero in FC layers, suggesting that the gradient-based sensitivity to geometric concepts is localised in the spatial feature extraction stages.

---

## 5. Methods and References

- **TCAV:** Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)*. ICML 2018.
- **Captum:** Kokhlikyan, N., et al. (2020). *Captum: A unified and generic model interpretability library for PyTorch*.
- **CAV spatial projection:** The heatmap method in Cells 20-21 extends standard TCAV by using the linear classifier weights as a projection operator onto spatially-resolved activation maps, enabling concept localisation without gradient computation.
