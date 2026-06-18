"""
Curvature Computation Experiments on Large Graphs

This experiment compares the mean curvature of different
curvature approximation methods on large graphs.

Large graphs:
- com-LiveJournal
- com-DBLP
- com-Amazon

Curvature approximation methods:
- CCOM
- MWPM
- LoCur
- ALU
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

import networkx as nx
from curvatures import compute_ccom, compute_mwpm, compute_locur, compute_alu
import gc
import numpy as np
import pandas as pd


def load_lj():
    """Load the com-LiveJournal dataset using only the top 3,750 highest-quality communities"""
    file_path = DATASETS_DIR / "com-lj/com-lj.ungraph.txt"
    cmty_path = DATASETS_DIR / "com-lj/com-lj.top5000.cmty.txt"

    # 1. Load the dataset
    G_full = nx.read_edgelist(
        path=file_path,
        comments='#',
        nodetype=int
    )
    G_full.remove_edges_from(nx.selfloop_edges(G_full))

    # 2. Load community information
    communities = []
    with open(cmty_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3750:
                break

            line = line.strip()
            if line:
                nodes = [int(n) for n in line.split()]
                communities.append(nodes)

    # 3. Select nodes from the top 3,750 highest-quality communities
    # and generate the induced subgraph
    community_nodes = set()
    for community in communities:
        community_nodes.update(community)
    G = G_full.subgraph(community_nodes).copy()
    del G_full
    gc.collect()

    return G


def load_amazon_or_dblp(dataset_name): # dataset_name:'amazon' or 'dblp'
    """Load the com-Amazon or com-DBLP dataset"""
    file_path = DATASETS_DIR / f"com-{dataset_name}/com-{dataset_name}.ungraph.txt"

    G = nx.read_edgelist(
        path=file_path,
        comments='#',
        nodetype=int
    )
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def load_roadnet_pa():
    """Load the roadNet-PA dataset"""
    file_path = DATASETS_DIR / "roadNet-PA.txt"

    G = nx.read_edgelist(
        path=file_path,
        comments='#',
        nodetype=int
    )
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def load_patents():
    """Load the cit-Patents dataset"""
    file_path = DATASETS_DIR / "cit-Patents.txt"

    G = nx.read_edgelist(
        path=file_path,
        comments='#',
        nodetype=int
    )
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def is_int_string(s):
    """Check if a string is an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def load_imdb():
    """Load the ca-IMDB dataset"""
    path = DATASETS_DIR / "IMDB.edges"

    G = nx.Graph()

    header_nodes = None
    header_edges = None

    raw_edge_lines = 0
    skipped_self_loops = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            # Handle comment lines or header lines
            if line.startswith("%"):
                parts = line[1:].strip().split()

                if len(parts) == 3 and all(is_int_string(p) for p in parts):
                    rows = int(parts[0])
                    cols = int(parts[1])
                    nnz = int(parts[2])

                    # To be consistent with Network Repository, use the second dimension as the number of nodes
                    header_nodes = cols
                    header_edges = nnz

                continue

            parts = line.split()

            if len(parts) == 3 and all(is_int_string(p) for p in parts):
                rows = int(parts[0])
                cols = int(parts[1])
                nnz = int(parts[2])

                if header_nodes is None:
                    header_nodes = cols
                    header_edges = nnz

                continue

            if len(parts) != 2:
                continue

            u, v = map(int, parts)

            if u == v:
                skipped_self_loops += 1
                continue

            G.add_edge(u, v)
            raw_edge_lines += 1

    # Explicitly add isolated nodes
    if header_nodes is not None:
        G.add_nodes_from(range(1, header_nodes + 1))

    return G


# ==================== Test Code ====================
if __name__ == "__main__":

    # ==================== 1. Load Dataset ====================
    print("Loading dataset...")

    # G = load_lj()
    G = load_amazon_or_dblp(dataset_name="amazon")  # dataset_name: "amazon" or "dblp"
    # G = load_imdb()
    # G = load_roadnet_pa()
    # G = load_patents()

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("Dataset loaded successfully.")

    # ==================== 2. Experiment Configuration ====================
    method_names = ["CCOM", "MWPM", "LoCur", "ALU"]

    method_to_attr_name = {
        "CCOM": "ccom",
        "MWPM": "mwpm",
        "LoCur": "locur",
        "ALU": "alu",
    }

    method_to_func = {
        "CCOM": compute_ccom,
        "MWPM": compute_mwpm,
        "LoCur": compute_locur,
        "ALU": compute_alu,
    }

    results = []  # Store the mean curvature of each approximation method

    # ==================== 3. Run Experiment ====================
    print("Starting curvature calculation experiment...")

    for method_name in method_names:
        print(f"\n========== Running {method_name} ==========")

        compute_func = method_to_func[method_name]
        attr_name = method_to_attr_name[method_name]

        G_result = compute_func(G)

        # Calculate mean curvature
        approx_curvatures = []
        for u, v, data in G_result.edges(data=True):
            approx_curvatures.append(data[attr_name])

        approx_curvatures = np.array(approx_curvatures, dtype=np.float64)
        mean = np.mean(approx_curvatures)

        results.append({
            "Methods": method_name,
            "Mean": mean,
        })

    # ==================== 4. Output Results ====================
    results_df = pd.DataFrame(results)

    print("\n========== Curvature Approximation Results ==========")
    print(results_df.to_string(index=False))