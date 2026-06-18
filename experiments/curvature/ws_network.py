"""
Investigate how the errors of different curvature approximation methods change
as the size of the WS network increases.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
EXACT_CURVATURES_DIR = DATA_DIR / "exact_curvatures"
FIGURES_DIR = PROJECT_ROOT / "figures"

import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from curvatures import compute_ccom, compute_mwpm, compute_locur, compute_alu, compute_exact_curvatures


# ==================== 1. Experiment Configuration ====================

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

# WS network parameters
k = 6
p = 0.1
seed = 42

# Node scale settings
sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 15000]

# Directory for saving TXT files of exact curvature values
exact_txt_dir = EXACT_CURVATURES_DIR / "ws"
os.makedirs(exact_txt_dir, exist_ok=True)

# Record the errors of each approximation method on WS networks of different scales
results = []


# ==================== 2. Run Experiments ====================

for size in sizes:
    print(f"\n========== Processing WS network: n={size} ==========")

    # Generate a WS network
    G = nx.watts_strogatz_graph(n=size, k=k, p=p, seed=seed)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

    # ---------- 2.1 Compute or load exact Ollivier-Ricci curvature ----------

    exact_txt_path = os.path.join(exact_txt_dir, f"ws_{size}.txt")
    exact_curvatures = []

    if os.path.exists(exact_txt_path):
        print(f"Loading exact ORC from {exact_txt_path}")

        with open(exact_txt_path, "r") as f:
            for line in f:
                exact_curvature = float(line.strip())
                exact_curvatures.append(exact_curvature)

    else:
        print("Computing exact Ollivier-Ricci curvature...")

        orc_G = compute_exact_curvatures(G)

        with open(exact_txt_path, "w") as f:
            for u, v in G.edges:
                exact_curvature = orc_G[u][v]["ricciCurvature"]
                print(exact_curvature, file=f)
                exact_curvatures.append(exact_curvature)

        print("Exact ORC saved to:")
        print(f"  TXT: {exact_txt_path}")

    # ---------- 2.2 Run each approximation method and compute errors ----------

    exact_curvatures = np.array(exact_curvatures, dtype=np.float64)

    # Count the number of zero-curvature edges in the exact curvature values
    zero_tol = 1e-12  # Edges with absolute exact curvature <= 1e-12 are treated as zero-curvature edges

    zero_curvature_mask = np.isclose(
        exact_curvatures,
        0.0,
        atol=zero_tol,
        rtol=0.0
    )

    nonzero_curvature_mask = ~zero_curvature_mask

    num_zero_curvature_edges = int(np.sum(zero_curvature_mask))
    zero_curvature_ratio = num_zero_curvature_edges / num_edges

    print(f"Zero-curvature edges: {num_zero_curvature_edges}")
    print(f"Zero-curvature edge ratio: {zero_curvature_ratio:.6f}")

    for method_name in method_names:
        print(f"Running method: {method_name}")

        compute_func = method_to_func[method_name]
        attr_name = method_to_attr_name[method_name]

        G_result = compute_func(G.copy())

        approx_curvatures = []
        for u, v, data in G_result.edges(data=True):
            approx_curvatures.append(data[attr_name])

        approx_curvatures = np.array(approx_curvatures, dtype=np.float64)

        # Calculate MAE
        mae = np.mean(np.abs(approx_curvatures - exact_curvatures))

        # Calculate MRE
        relative_errors = (
                np.abs(
                    approx_curvatures[nonzero_curvature_mask]
                    - exact_curvatures[nonzero_curvature_mask]
                )
                / np.abs(exact_curvatures[nonzero_curvature_mask])
        )

        mre = np.mean(relative_errors)

        result_record = {
            "network": "WS",
            "n": size,
            "k": k,
            "p": p,
            "seed": seed,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_zero_curvature_edges": num_zero_curvature_edges,
            "zero_curvature_ratio": zero_curvature_ratio,
            "method": method_name,
            "MAE": mae,
            "MRE": mre,
        }

        results.append(result_record)


# ==================== 3. Experimental Results ====================

results_df = pd.DataFrame(results)


# ==================== 4. Plot and Save Line Charts ====================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})

method_markers = {
    "CCOM": "o",
    "MWPM": "s",
    "LoCur": "^",
    "ALU": "D",
}

method_linestyles = {
    "CCOM": "-",
    "MWPM": "--",
    "LoCur": "-.",
    "ALU": ":",
}


def plot_error_curve(
    results_df,
    metric_name,
    ylabel,
    title,
    save_filename,
    legend_loc="best",
    legend_bbox=None,
    y_as_percent=False
):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for method_name in method_names:
        sub_df = results_df[results_df["method"] == method_name].sort_values("n")

        ax.plot(
            sub_df["n"],
            sub_df[metric_name],
            label=method_name,
            marker=method_markers.get(method_name, "o"),
            linestyle=method_linestyles.get(method_name, "-"),
            linewidth=2.0,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_xscale("log")

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if y_as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6
    )

    if legend_bbox is None:
        ax.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black",
            loc=legend_loc
        )
    else:
        ax.legend(
            frameon=True,
            fancybox=False,
            edgecolor="black",
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            borderaxespad=0.5
        )

    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=30)

    fig.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print("Figure saved to:")
    print(f"  PNG: {fig_path}")


# ---------- 4.1 Plot the MAE curve ----------
plot_error_curve(
    results_df=results_df,
    metric_name="MAE",
    ylabel="Mean Absolute Error",
    title="Curvature Approximation Error on WS Networks",
    save_filename="ws_curvature_approximation_mae.png",
    legend_loc="lower right",
    legend_bbox=(1.0, 0.08)
)

# ---------- 4.2 Plot the MRE curve ----------
plot_error_curve(
    results_df=results_df,
    metric_name="MRE",
    ylabel="Mean Relative Error",
    title="Curvature Approximation Error on WS Networks",
    save_filename="ws_curvature_approximation_mre.png",
    legend_loc="lower right",
    legend_bbox=(1.0, 0.08),
    y_as_percent=True
)


# ==================== 5. Output Results ====================

print("\n========== Results ==========")
print(results_df.to_string(index=False))
