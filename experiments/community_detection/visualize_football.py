"""
Visualize the community detection results on the Football dataset.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
FIGURES_DIR = PROJECT_ROOT / "figures"

import os
from collections import defaultdict
import networkx as nx
from curvature_community_detection import CurvatureBasedCD
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


# ==================== 1. Basic Utility Functions ====================

def group_nodes_by_feature(G, feature_name):
    """Group nodes by a given node feature."""
    groups = defaultdict(list)

    for node, data in G.nodes(data=True):
        val = data.get(feature_name)
        if val is not None:
            groups[val].append(node)

    return groups


def align_predicted_labels(gt_labels, pred_labels):
    """Align predicted labels to ground-truth labels."""
    unique_gt = np.unique(gt_labels)
    unique_pred = np.unique(pred_labels)

    map_gt = {l: i for i, l in enumerate(unique_gt)}
    map_pred = {l: i for i, l in enumerate(unique_pred)}

    matrix = np.zeros((len(unique_gt), len(unique_pred)), dtype=int)

    for g, p in zip(gt_labels, pred_labels):
        matrix[map_gt[g]][map_pred[p]] += 1

    row_ind, col_ind = linear_sum_assignment(-matrix)

    translation_dict = {}
    used_pred_labels = set()

    gt_vals = list(unique_gt)
    pred_vals = list(unique_pred)

    for r, c in zip(row_ind, col_ind):
        original_pred = pred_vals[c]
        target_gt = gt_vals[r]
        translation_dict[original_pred] = target_gt
        used_pred_labels.add(original_pred)

    next_id = max(gt_labels) + 1

    for p in unique_pred:
        if p not in used_pred_labels:
            translation_dict[p] = next_id
            next_id += 1

    return [translation_dict.get(p, p) for p in pred_labels]


# ==================== 2. Adaptive Circle Layout ====================

def get_adaptive_circle_layout(groups):
    """Generate an adaptive circular layout for communities."""
    pos = {}

    sorted_keys = sorted(groups.keys())
    num_groups = len(sorted_keys)

    R_big = 22.0

    for i, key in enumerate(sorted_keys):
        nodes = sorted(groups[key])
        n_nodes = len(nodes)

        if n_nodes == 0:
            continue

        # Adaptive radius
        r_dynamic = max(2.0, n_nodes * 0.38)

        angle_big = 2 * math.pi * i / num_groups
        center_x = R_big * math.cos(angle_big)
        center_y = R_big * math.sin(angle_big)

        for j, node in enumerate(nodes):
            angle_small = 2 * math.pi * j / n_nodes

            px = center_x + r_dynamic * math.cos(angle_small)
            py = center_y + r_dynamic * math.sin(angle_small)

            pos[node] = (px, py)

    return pos


# ==================== 3. Main Visualization Function ====================

def visualize_football_comparison(path_to_gml=DATASETS_DIR / "football.gml"):
    """Visualize Football community detection results."""
    # 1. Create the output directory
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 2. Load the graph
    try:
        G = nx.read_gml(path_to_gml)
        print(f"Successfully loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except FileNotFoundError:
        print(f"Error: file not found {path_to_gml}")
        return

    # 3. Build ground-truth communities
    groups = group_nodes_by_feature(G, "value")

    if not groups:
        print("Error: node attribute 'value' not found, cannot build ground-truth communities.")
        return

    num_of_communities = len(groups)
    min_community_size = min([len(c) for c in groups.values()]) if groups else 1
    chunk_size = max(G.number_of_edges() // 100, 1)

    node_to_id = {}
    sorted_keys = sorted(groups.keys())

    for idx, label_key in enumerate(sorted_keys):
        for node in groups[label_key]:
            node_to_id[node] = idx

    nodes_list = list(G.nodes())
    gt_flat_labels = [node_to_id.get(n, 0) for n in nodes_list]

    # 4. Use an adaptive layout
    print("Generating adaptive circle layout...")
    pos = get_adaptive_circle_layout(groups)

    # 5. Define a helper function
    def run_and_align(curvature_name):
        """Run detection and align labels."""
        print(f"Running {curvature_name}...", end="")

        try:
            cd = CurvatureBasedCD(
                G,
                curvature_name=curvature_name,
                number_of_communities=num_of_communities,
                minimum_community_size=min_community_size,
                chunk_size=chunk_size
            )

            detected_coms_list = cd.detect()

            pred_map = {}

            for cid, comm in enumerate(detected_coms_list):
                for node in comm:
                    pred_map[node] = cid

            raw_pred_labels = [pred_map.get(n, 0) for n in nodes_list]
            aligned_labels = align_predicted_labels(gt_flat_labels, raw_pred_labels)

            print(" Done.")

            return aligned_labels

        except Exception as e:
            print(f" Failed: {e}")
            return [0] * len(nodes_list)

    # 6. Run community detection
    methods = {
        "Ground Truth": gt_flat_labels,
        "CCOM": run_and_align("CCOM"),
        "Others": run_and_align("LRC")  # Use LRC as a representative of other methods
    }

    plot_sequence = [
        "Ground Truth",
        "CCOM",
        "Others"
    ]

    # 7. Plot settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 1.0,
        "mathtext.fontset": "stix",
        "mathtext.default": "it",
    })

    # Use a 1x3 layout
    fig = plt.figure(figsize=(21, 7), dpi=300)

    gs = fig.add_gridspec(
        1,
        3,
        wspace=0.12
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    axes_list = [ax1, ax2, ax3]


    cmap = plt.cm.tab20

    # Use a shared color range for all subplots
    all_labels = []
    for method_name in plot_sequence:
        labels = methods[method_name]
        all_labels.extend(labels)

    vmax = max(19, max(all_labels) if all_labels else 0)

    # 8. Draw subplots
    for i, method_name in enumerate(plot_sequence):
        ax = axes_list[i]

        labels = methods[method_name]

        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.1,
            edge_color="gray",
            width=0.6,
            ax=ax
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=labels,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            node_size=160,
            edgecolors="black",
            linewidths=0.8,
            ax=ax
        )

        # Draw node labels
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=7,
            font_color="black",
            font_family="sans-serif",
            ax=ax
        )

        ax.set_aspect("equal")
        ax.axis("off")

        # Place subplot titles below
        letter = chr(97 + i)
        formula_letter = rf"$({letter})$"

        if method_name == "Ground Truth":
            subtitle = rf"{formula_letter} {method_name}"
        elif method_name == "CCOM":
            subtitle = rf"{formula_letter} {method_name}"
        else:
            subtitle = rf"{formula_letter} Others (FRC, BFC, LoCur, etc.)"

        title_box = HPacker(
            children=[
                TextArea(
                    formula_letter,
                    textprops={
                        "fontsize": 28,
                        "fontweight": "bold",
                    }
                ),
                TextArea(
                    subtitle.replace(formula_letter, ""),
                    textprops={
                        "fontsize": 28,
                        "fontfamily": "Times New Roman",
                        "fontweight": "normal",
                    }
                ),
            ],
            align="center",
            pad=0,
            sep=6,
        )

        anchored_title = AnchoredOffsetbox(
            loc="lower center",
            child=title_box,
            pad=0,
            frameon=False,
            bbox_to_anchor=(0.5, -0.12),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        ax.add_artist(anchored_title)

    # 9. Save the figure
    plt.subplots_adjust(
        hspace=0.18,
        wspace=0.08
    )

    save_path = os.path.join(FIGURES_DIR, "football_communities_comparison.png")

    plt.savefig(
        save_path,
        format="png",
        bbox_inches="tight"
    )

    print(f"Figure saved to: {save_path}")

    plt.show()


# ==================== Football community detection visualization ====================
visualize_football_comparison(DATASETS_DIR / "football.gml")

