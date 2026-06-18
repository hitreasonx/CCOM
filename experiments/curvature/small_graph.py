"""
Curvature Computation Experiments on Small Graphs and Medium-Sized Networks

This experiment compares the error of different curvature approximation
methods on small graphs and medium-sized networks.

Small graphs:
- Florentine families
- Karate
- DD graph No. 780
- Dolphin
- Les miserables
- Polbooks
- Football
- REDDIT-5K graph No. 143

Medium-sized networks:
- Cora
- Citeseer
- PubMed

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
EXACT_CURVATURES_DIR = DATA_DIR / "exact_curvatures"

from curvatures import compute_ccom, compute_mwpm, compute_locur, compute_alu, compute_exact_curvatures
import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.utils as pyg_utils
import pandas as pd
import os


def manual_to_nx(data):
    """Manually convert DD and REDDIT-Multi-5K samples to NetworkX graphs"""
    # 1. Create an empty undirected graph
    G = nx.Graph()

    # 2. Add nodes indexed from 0 to N-1
    G.add_nodes_from(range(data.num_nodes))

    # 3. Add edges
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    return G


# ==================== Test Code ====================
if __name__ == "__main__":
    # ==================== 1. Load Dataset ====================
    print("Loading dataset...")

    # ---------- Karate / Les miserables / Florentine families ----------
    G = nx.karate_club_graph()
    exact_txt_path = EXACT_CURVATURES_DIR / "karate_club.txt"  # Path to the TXT file storing exact curvature values

    # G = nx.les_miserables_graph()
    # exact_txt_path = EXACT_CURVATURES_DIR / "les_miserables.txt"

    # G = nx.florentine_families_graph()
    # exact_txt_path = EXACT_CURVATURES_DIR / "florentine_families.txt"


    # ---------- Dolphin / Polbooks / Football ----------
    # name = 'dolphins' # name: 'dolphins', 'polbooks' or 'football'
    # exact_txt_path = EXACT_CURVATURES_DIR / f"{name}.txt"


    # ---------- DD graph No. 780 ----------
    # dataset = TUDataset(root=str(DATASETS_DIR), name='DD')
    # G = manual_to_nx(dataset[780])
    # exact_txt_path = EXACT_CURVATURES_DIR / "dd.txt"


    # ---------- REDDIT-5K graph No. 143 ----------
    # dataset = TUDataset(root=str(DATASETS_DIR), name='REDDIT-MULTI-5K')
    # G = manual_to_nx(dataset[143])
    # exact_txt_path = EXACT_CURVATURES_DIR / "reddit-multi-5k.txt"


    # ---------- Cora / Citeseer / PubMed ----------
    # name = "PubMed"  # name: 'Cora', 'Citeseer' or 'PubMed'
    # dataset = Planetoid(root=str(DATASETS_DIR), name=name)
    # data = dataset[0]
    # G = pyg_utils.to_networkx(data, to_undirected=True)
    # G.remove_edges_from(nx.selfloop_edges(G))
    # exact_txt_path = EXACT_CURVATURES_DIR / f"{name.lower()}.txt"

    print("Dataset loaded successfully.")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

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

    # Compute or load exact curvature values. For convenience, we provide TXT files
    # containing precomputed exact curvature values, but they can also be computed
    # using GraphRicciCurvature.
    exact_curvatures = []  # Store exact curvature values

    if os.path.exists(exact_txt_path):
        print(f"Loading exact ORC from {exact_txt_path}")

        f = open(exact_txt_path)
        for line in f.readlines():
            exact_curvature = float(line.strip())
            exact_curvatures.append(exact_curvature)
        f.close()

    else:
        print("Computing exact Ollivier-Ricci curvature...")

        orc_G = compute_exact_curvatures(G)

        with open(exact_txt_path, "w") as f:
            for u, v in G.edges:
                exact_curvature = orc_G[u][v]['ricciCurvature']
                print(exact_curvature, file=f)
                exact_curvatures.append(exact_curvature)

        print(f"Exact ORC saved to:")
        print(f"  TXT: {exact_txt_path}")

    # Store MAE, MRE, and mean curvature for each approximation method
    results = []

    # ==================== 3. Run Experiment ====================
    print("Starting curvature calculation experiment...")

    for method_name in method_names:
        print(f"\n========== Running {method_name} ==========")

        compute_func = method_to_func[method_name]
        attr_name = method_to_attr_name[method_name]

        print(f"Run the algorithm and calculate MAE, MRE and Mean for {method_name} ...")

        G_result = compute_func(G)

        # Calculate MAE, MRE, and mean curvature
        approx_curvatures = []
        for u, v, data in G_result.edges(data=True):
            approx_curvatures.append(data[attr_name])

        approx_curvatures = np.array(approx_curvatures, dtype=np.float64)
        exact_curvatures = np.array(exact_curvatures, dtype=np.float64)

        # Calculate MAE
        mae = np.mean(np.abs(approx_curvatures - exact_curvatures))

        # Calculate MRE
        zero_tol = 1e-12  # Edges with absolute exact curvature <= 1e-12 are treated as zero-curvature edges

        zero_curvature_mask = np.isclose(
            exact_curvatures,
            0.0,
            atol=zero_tol,
            rtol=0.0
        )

        nonzero_curvature_mask = ~zero_curvature_mask

        relative_errors = (
                np.abs(
                    approx_curvatures[nonzero_curvature_mask]
                    - exact_curvatures[nonzero_curvature_mask]
                )
                / np.abs(exact_curvatures[nonzero_curvature_mask])
        )

        mre = np.mean(relative_errors)

        # Calculate mean curvature
        mean = np.mean(approx_curvatures)

        results.append({
            "Methods": method_name,
            "MAE": mae,
            "MRE": mre,
            "Mean": mean,
        })


    # ==================== 4. Output Results ====================
    results.append({
        "Methods": "Exact ORC",
        "MAE": None,
        "MRE": None,
        "Mean": np.mean(exact_curvatures),
    })
    results_df = pd.DataFrame(results)

    print("\n========== Curvature Approximation Results ==========")
    print(results_df.to_string(index=False))
