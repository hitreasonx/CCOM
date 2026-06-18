# Reference:
# The BFC implementation in this file is adapted from the authors' implementation for:
# Park, Y. J., & Li, D. (2025). Lower Ricci curvature for efficient community detection.
# Transactions on Machine Learning Research, pp. 3701.

import networkx as nx
from .bfc_numba import balanced_forman_curvature


class BFC:
    """Compute Balanced Forman curvature for graph edges.

    This class computes Balanced Forman curvature for a given NetworkX graph.
    The computed values are stored as edge attributes named ``"bfc"``.
    """
    def __init__(self, G: nx.Graph):
        """Initialize the Balanced Forman curvature calculator.

        Args:
            G (networkx.Graph): The input graph.
        """
        self.G = G.copy()

    def set_balancedforman_edge(self, C):
        """Make a dictionary called attri that saves the value of BFC for each edge
        Parameters
        ----------
        C: The curvature matrix.

        Returns
        -------
        attri: A dictionary that saves that value of BFC as value and edge as keys.
        """
        attri = {}
        nodelist = list(self.G)

        for i, j in nx.edges(self.G):
            x = nodelist.index(i)
            y = nodelist.index(j)
            attri[(i, j)] = C[x, y]
            attri[(j, i)] = C[y, x]

        return attri

    def compute_balancedformancurv(self):
        """Compute Balanced Forman Curvature of edges.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "bfc" on edges.
        """
        A = nx.to_numpy_array(self.G)
        curv = balanced_forman_curvature(A, C=None)
        a = self.set_balancedforman_edge(curv)
        nx.set_edge_attributes(self.G, a, "bfc")

        return self.G

