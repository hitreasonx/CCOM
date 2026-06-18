"""
Evaluate curvature-based community detection on small graphs.

Datasets: Dolphin, Polbooks, Football.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

import networkx as nx
from collections import defaultdict
from cdlib import evaluation, NodeClustering
from curvature_community_detection import CurvatureBasedCD
import pandas as pd


def load_dataset(dataset_name):
    """Load a GML dataset and its ground-truth communities."""
    G = nx.read_gml(DATASETS_DIR / f"{dataset_name}.gml", label="id")

    groups = defaultdict(list)
    for node, data in G.nodes(data=True):
        val = data.get("value")
        groups[val].append(node)

    communities = list(groups.values())

    return G, communities


def compute_metrics(ground_truth, detected_coms):
    """Compute ARI and AMI scores."""
    ari = evaluation.adjusted_rand_index(ground_truth, detected_coms).score
    ami = evaluation.adjusted_mutual_information(ground_truth, detected_coms).score

    return ari, ami


# ==================== Test Code ====================
# ==================== 1. Load Dataset and Ground-Truth Communities ====================
print("Loading dataset and ground truth communities...")

dataset_name = "polbooks" # name: "dolphins", "polbooks" or "football"
G, communities = load_dataset(dataset_name)

print("Loading success!")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

num_of_communities = len(communities)
min_community_size = min(len(comm) for comm in communities)
chunk_size = max(G.number_of_edges() // 100, 1)

print(f"Number of Communities: {num_of_communities}")
print(f"Minimum community size: {min_community_size}")
print(f"Chunk size： {chunk_size}")

# Wrap communities in NodeClustering format
gt_coms = NodeClustering(communities, graph=G, method_name="GroundTruth")


# ==================== 2. Run Curvature-Based Community Detection ====================
results = []
curvatures_to_test = ['LRC', 'FRC', 'BFC', 'LoCur', 'ALU', 'MWPM', 'CCOM']
for curvature_name in curvatures_to_test:
    cd = CurvatureBasedCD(G,
                          curvature_name = curvature_name,
                          number_of_communities = num_of_communities,
                          minimum_community_size = min_community_size,
                          chunk_size = chunk_size)

    detected_coms = cd.detect()
    detected_coms = NodeClustering(detected_coms, graph=G, method_name=curvature_name)

    # Compute metrics
    ari, ami = compute_metrics(gt_coms, detected_coms)
    results.append({
        "Curvatures": curvature_name,
        "ARI": ari,
        "AMI": ami,
    })

# ==================== 3. Output Results ====================
results_df = pd.DataFrame(results)

print("\n========== Community Detection Results ==========")
print(results_df.to_string(index=False))



