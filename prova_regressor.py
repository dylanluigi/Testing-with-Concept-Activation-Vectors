import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from captum.concept import TCAV, Concept
from captum.concept._utils.classifier import DefaultClassifier
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. SETUP: Dummy Model and Data
# ==========================================
# Use weights=None to avoid warnings, or use 'DEFAULT' if you have internet
model = models.resnet18(weights=None) 
model.eval()

target_layer = 'layer4'

print("Generating dummy data...")
BATCH_SIZE = 10
INPUT_SHAPE = (3, 224, 224)

# Create Dummy Data
stripes_data = torch.randn(50, *INPUT_SHAPE)
random_data = torch.randn(50, *INPUT_SHAPE)

def get_dataloader(data):
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=BATCH_SIZE)

stripes_loader = get_dataloader(stripes_data)
random_loader = get_dataloader(random_data)

# ==========================================
# 2. CAPTUM TCAV EXECUTION
# ==========================================
print(f"Running TCAV on {target_layer}...")

# Define Concepts
concept_stripes = Concept(id=0, name="stripes", data_iter=stripes_loader)
concept_random = Concept(id=1, name="random", data_iter=random_loader)

# Initialize TCAV
tcav = TCAV(model=model, layers=[target_layer])

# CRITICAL STEP: Explicitly compute CAVs first to ensure they are accessible
# This returns the dictionary {Concept_Object: {Layer: Stats}}
print("Computing CAVs explicitly...")
cav_results = tcav.compute_cavs(experimental_sets=[[concept_stripes, concept_random]])

# Compute Scores (Optional, just to finish the TCAV workflow)
scores = tcav.interpret(
    inputs=stripes_data[:5],
    experimental_sets=[[concept_stripes, concept_random]],
    target=0
)
print("TCAV Score calculated.")

# ==========================================
# 3. EXTRACTING WEIGHTS (The Fix)
# ==========================================
print("Extracting Classifier Weights (CAV)...")

# cav_results keys are concept-pair strings like '0-1' (from concept IDs)
print(f"Keys found: {list(cav_results.keys())}")
# Use the first available key (there's only one experimental set)
cav_key = list(cav_results.keys())[0]
cav_data = cav_results[cav_key]

# Handle the Layer Key (String vs Module Object)
# We just take the first available layer key
layer_key = list(cav_data.keys())[0]
cav_obj = cav_data[layer_key]  # This is a CAV object

# Extract weights from the CAV stats dict
# stats['weights'] has shape [2, N_channels] — row 0 is concept_stripes (id=0)
raw_weights = cav_obj.stats['weights']

# Convert to Numpy
if torch.is_tensor(raw_weights):
    weights_np = raw_weights.detach().cpu().numpy()
else:
    weights_np = np.array(raw_weights)

# Take only the stripes concept row (index 0), shape becomes [N_channels]
weights_np = weights_np[0]

print(f"Weights extracted successfully. Shape: {weights_np.shape}")

# ==========================================
# 4. PLOTTING
# ==========================================
# If we have 512 weights, we can plot channel importance
num_channels = weights_np.shape[0]

# Simple check to ensure we have a 1D vector (Channel Importance)
if len(weights_np.shape) == 1:
    top_n = 20
    sorted_indices = np.argsort(np.abs(weights_np))[::-1][:top_n]
    top_weights = weights_np[sorted_indices]

    plt.figure(figsize=(12, 6))
    colors = ['red' if w < 0 else 'blue' for w in top_weights]
    plt.bar(range(top_n), top_weights, color=colors)
    plt.xticks(range(top_n), sorted_indices, rotation=45)
    plt.xlabel(f"Channel Index (in {target_layer})")
    plt.ylabel("Weight Magnitude")
    plt.title(f"Top {top_n} Channels for 'Stripes' (Concept Object Key)")
    plt.legend(["Blue = Pro-Concept, Red = Anti-Concept"])
    plt.tight_layout()
    plt.show()
    print("Plot generated.")
else:
    print(f"Complex weight shape detected ({weights_np.shape}). Requires manual reshaping logic.")