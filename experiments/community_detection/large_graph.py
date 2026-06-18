"""
Evaluate curvature-based community detection on large graphs.

Datasets: com-LiveJournal, com-DBLP, com-Amazon.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

import networkx as nx
import gc
from collections import defaultdict
from cdlib import evaluation, NodeClustering, readwrite
from curvature_community_detection import CurvatureBasedCD
import pandas as pd


def load_dataset_and_get_top_number_ground_truth(dataset_name, number): # dataset_name:'amazon', 'dblp' or 'lj'
    """Load a dataset and the top-number ground-truth communities."""
    file_path = DATASETS_DIR / f"com-{dataset_name}/com-{dataset_name}.ungraph.txt"
    cmty_path = DATASETS_DIR / f'com-{dataset_name}/com-{dataset_name}.top5000.cmty.txt'

    # 1. Load the dataset
    G = nx.read_edgelist(
        path=file_path,
        comments='#',
        nodetype=int
    )
    G.remove_edges_from(nx.selfloop_edges(G))

    # 2. Load communities
    communities = []
    with open(cmty_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= number:
                break

            line = line.strip()
            if line:
                nodes = [int(n) for n in line.split()]
                communities.append(nodes)

    return G, communities


def compute_f1_score_for_gt_nodes_only(ground_truth, detected_coms, gt_nodes=None):
    """Compute F1-score using only nodes with ground-truth labels."""
    if gt_nodes is not None:
        filtered_coms = []
        for community in detected_coms.communities:
            filtered_comm = list(set(community) & gt_nodes)
            if len(filtered_comm) >= 1:
                filtered_coms.append(filtered_comm)

        if filtered_coms:
            filtered_detected_coms = NodeClustering(
                filtered_coms,
                graph=detected_coms.graph,
                method_name=detected_coms.method_name,
                overlap=detected_coms.overlap
            )
        else:
            return 0.0
    else:
        filtered_detected_coms = detected_coms

    f1 = evaluation.f1(ground_truth, filtered_detected_coms).score
    return f1


def run_experiment_on_lj():
    """Run the experiment on the com-LiveJournal dataset."""
    # 1. Load the dataset and ground truth communities
    print("Loading dataset and ground truth communities...")
    G_full, communities = load_dataset_and_get_top_number_ground_truth(dataset_name="lj", number=3750)

    community_nodes = set()
    for community in communities:
        community_nodes.update(community)
    G = G_full.subgraph(community_nodes).copy()
    del G_full
    gc.collect()

    print("Loading success!")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # 2. Set experiment parameters and wrap ground-truth communities
    number_of_communities = None
    minimum_community_size = None
    chunk_size = max(G.number_of_edges() // 100, 1)
    gt_coms = NodeClustering(communities, graph=G, method_name="Ground Truth", overlap=True)

    # 3. Run curvature-based community detection
    results = []
    curvatures_to_test = ['LRC', 'FRC', 'LoCur', 'ALU', 'MWPM', 'CCOM']
    for curvature_name in curvatures_to_test:
        cd = CurvatureBasedCD(G,
                              curvature_name=curvature_name,
                              number_of_communities=number_of_communities,
                              minimum_community_size=minimum_community_size,
                              chunk_size=chunk_size)

        detected_coms = cd.detect()
        detected_coms = NodeClustering(detected_coms, graph=G, method_name=curvature_name, overlap=True)

        # Compute metrics
        print(f"Metrics for {curvature_name}:")
        f1_score = evaluation.f1(gt_coms, detected_coms).score
        print(f"F1-score : {f1_score}")

        results.append({
            "Curvatures": curvature_name,
            "F1-score": f1_score,
        })

    # 4. Output results
    results_df = pd.DataFrame(results)

    print("\n========== Community Detection Results ==========")
    print(results_df.to_string(index=False))


def run_experiment_on_dblp_or_amazon(dataset_name): # dataset_name: 'dblp' or 'amazon'
    """Run the experiment on the com-DBLP or com-Amazon dataset."""
    # 1. Load the dataset
    print("Loading dataset and ground truth communities...")
    G, communities = load_dataset_and_get_top_number_ground_truth(dataset_name=dataset_name, number=5000)

    print("Loading success!")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # 2. Set experiment parameters and wrap ground-truth communities
    number_of_communities = None
    minimum_community_size = None
    chunk_size = max(G.number_of_edges() // 100, 1)
    gt_coms = NodeClustering(communities, graph=G, method_name="Ground Truth", overlap=True)
    gt_nodes = set(gt_coms.to_node_community_map().keys())

    # 3. Run curvature-based community detection
    results = []
    curvatures_to_test = ['LRC', 'FRC', 'LoCur', 'ALU', 'MWPM', 'CCOM']
    for curvature_name in curvatures_to_test:
        cd = CurvatureBasedCD(G,
                              curvature_name=curvature_name,
                              number_of_communities=number_of_communities,
                              minimum_community_size=minimum_community_size,
                              chunk_size=chunk_size)

        detected_coms = cd.detect()
        detected_coms = NodeClustering(detected_coms, graph=G, method_name=curvature_name, overlap=True)

        # Compute metrics
        print(f"Metrics for {curvature_name}:")
        f1_score = compute_f1_score_for_gt_nodes_only(gt_coms, detected_coms, gt_nodes)
        print(f"F1-score : {f1_score}")

        results.append({
            "Curvatures": curvature_name,
            "F1-score": f1_score,
        })

    # 4. Output results
    results_df = pd.DataFrame(results)

    print("\n========== Community Detection Results ==========")
    print(results_df.to_string(index=False))



# ==================== Test Code ====================
run_experiment_on_lj()

# run_experiment_on_dblp_or_amazon("dblp")

# run_experiment_on_dblp_or_amazon("amazon")



