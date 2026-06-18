"""
To measure running time and peak memory usage on Linux, run:
/usr/bin/time -v python experiments/curvature/time_mem.py

Check the field:
Maximum resident set size (kbytes)
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

import networkx as nx
import time
import torch
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.utils as pyg_utils
import gc
from small_graph import manual_to_nx
from large_graph import load_lj, load_amazon_or_dblp, load_imdb, load_roadnet_pa, load_patents
from curvatures import compute_ccom, compute_mwpm, compute_locur, compute_alu, compute_exact_curvatures

# ==================== Test Code ====================
# ==================== 1. Load Dataset ====================
print("Loading dataset...")

# ---------- Karate / Les miserables / Florentine families ----------
# G = nx.karate_club_graph()
# G = nx.les_miserables_graph()
# G = nx.florentine_families_graph()

# ---------- Dolphin / Football / Polbooks ----------
# name = 'dolphins' # name: 'dolphins', 'polbooks' or 'football'
# G = nx.read_gml(DATASETS_DIR / f"{name}.gml", label="id")

# ---------- DD graph No. 780 ----------
# dataset = TUDataset(root=str(DATASETS_DIR), name='DD')
# G = manual_to_nx(dataset[780])

# ---------- REDDIT-5K graph No. 143 ----------
# dataset = TUDataset(root=str(DATASETS_DIR), name='REDDIT-MULTI-5K')
# G = manual_to_nx(dataset[143])

# ---------- Cora / Citeseer / PubMed ----------
# name = "PubMed"  # name: 'Cora', 'Citeseer' or 'PubMed'
# dataset = Planetoid(root=str(DATASETS_DIR), name=name)
# data = dataset[0]
# G = pyg_utils.to_networkx(data, to_undirected=True)
# G.remove_edges_from(nx.selfloop_edges(G))

# ---------- com-LiveJournal ----------
# G = load_lj()

# ---------- com-Amazon / com-DBLP ----------
G = load_amazon_or_dblp(dataset_name="amazon") # dataset_name: "amazon" or "dblp"

# ---------- IMDB ----------
# G = load_imdb()

# ---------- roadNet-PA ----------
# G = load_roadnet_pa()

# ---------- cit-Patents ----------
# G = load_patents()

print("Dataset loaded successfully.")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# ==================== 2. Run Experiment ====================
start = time.perf_counter()
# Replace different functions to obtain the time and memory usage of
# the corresponding curvature approximation methods and Exact ORC. Such as compute_ccom,
# compute_mwpm, compute_locur, compute_alu and compute_exact_curvatures
G_result = compute_ccom(G)
end = time.perf_counter()
print(f"Running time: {end - start:.6f} seconds.")

