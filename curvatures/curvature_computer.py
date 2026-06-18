"""Utility functions for computing graph curvature measures.

This module provides lightweight wrapper functions for different curvature
measures. Each function takes a NetworkX graph as input and returns a copied
graph with the corresponding curvature values stored as edge attributes.
"""

import networkx as nx
from .ccom import CCOM
from .mwpm import MWPM
from .locur import LoCur
from .alu import ALU
from .frc import FRC
from .bfc import BFC
from .lrc import LRC
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def compute_ccom(G: nx.Graph):
    """Compute approximate curvature using CCOM for all edges."""
    ccom = CCOM(G)
    G = ccom.calculate_all_edges_ccom()
    return G


def compute_mwpm(G: nx.Graph):
    """Compute approximate curvature using MWPM for all edges."""
    mwpm = MWPM(G)
    G = mwpm.calculate_all_edges_curvature()
    return G


def compute_locur(G: nx.Graph):
    """Compute approximate curvature using LoCur for all edges."""
    locur = LoCur(G)
    G = locur.calculate_all_edges_locur()
    return G


def compute_alu(G: nx.Graph):
    """Compute approximate curvature using ALU for all edges."""
    alu = ALU(G)
    G = alu.calculate_all_edges_alu()
    return G


def compute_frc(G: nx.Graph):
    """Compute Forman Ricci curvature for all edges."""
    frc = FRC(G)
    G = frc.forman_curvature()
    return G


def compute_bfc(G: nx.Graph):
    """Compute Balanced Forman curvature for all edges."""
    bfc = BFC(G)
    G = bfc.compute_balancedformancurv()
    return G


def compute_lrc(G: nx.Graph):
    """Compute Lower Ricci curvature for all edges."""
    lrc = LRC(G)
    G = lrc.calculate_all_edges_lrc()
    return G


def compute_exact_curvatures(G: nx.Graph):
    """Compute Exact ORC for all edges."""
    orc = OllivierRicci(
        G,
        alpha=0,
        verbose="INFO",
        method="OTD",
        weight=None,
        shortest_path="all_pairs"
    )

    orc.compute_ricci_curvature()

    return orc.G