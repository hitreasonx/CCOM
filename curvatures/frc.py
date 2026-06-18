# Reference:
# The FRC implementation in this file is adapted from the authors' implementation for:
# Park, Y. J., & Li, D. (2025). Lower Ricci curvature for efficient community detection.
# Transactions on Machine Learning Research, pp. 3701.


import networkx as nx

class FRC:
    """Compute Forman-Ricci curvature for graph edges.

    This class computes the Forman-Ricci curvature for each edge of a given
    NetworkX graph. The computed curvature values are stored as edge attributes
    named ``"frc"``.
    """
    def __init__(self, G: nx.Graph):
        """Initialize the Forman-Ricci curvature calculator.

        Args:
            G (networkx.Graph): The input graph.
        """
        self.G = G.copy()

    def forman_curvature(self):
        """Compute Forman-Ricci curvature for all edges.

        For each edge ``(i, j)``, the Forman-Ricci curvature is computed using
        the degrees of the two endpoints and the number of their common
        neighbors. The computed values are stored as edge attributes named
        ``"frc"``.

        Returns:
            networkx.Graph: A copy of the input graph with Forman-Ricci
            curvature values stored as edge attributes named ``"frc"``.
        """
        C = {}
        for i, j in self.G.edges:
            n_ij = len(sorted(nx.common_neighbors(self.G, i, j)))
            n_i = len(sorted(nx.all_neighbors(self.G, i)))
            n_j = len(sorted(nx.all_neighbors(self.G, j)))
            C[(i, j)] = 4 - n_i - n_j + 3 * n_ij

        nx.set_edge_attributes(self.G, C, "frc")
        return self.G