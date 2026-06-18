# Reference:
# The LRC implementation in this file is adapted from the authors' implementation for:
# Park, Y. J., & Li, D. (2025). Lower Ricci curvature for efficient community detection.
# Transactions on Machine Learning Research, pp. 3701.

import networkx as nx


class LRC:
    """Compute LRC for all edges in a graph.

    This class takes a NetworkX graph, creates an internal copy, and assigns
    an ``'lrc'`` attribute to each edge based on the LRC formula.
    """

    def __init__(self, G: nx.Graph):
        """Initialize the LRC calculator with a copy of the input graph.

        Args:
            G (nx.Graph): The input graph.
        """
        self.G = G.copy()

    def calculate_all_edges_lrc(self):
        """Compute LRC for every edge in the graph.

        For each edge, this method computes the number of common neighbors
        and the degrees of its two endpoints. The resulting LRC
        values are stored as edge attributes named ``'lrc'``.

        Returns:
            nx.Graph: A copy of the input graph with LRC values
            stored in edge attributes.
        """
        C = {}

        for i, j in self.G.edges:
            # Number of common neighbors between the two endpoints.
            n_ij = len(sorted(nx.common_neighbors(self.G, i, j)))

            # Degrees of the two endpoints.
            n_i = len(sorted(nx.all_neighbors(self.G, i)))
            n_j = len(sorted(nx.all_neighbors(self.G, j)))

            n_max = max(n_i, n_j)
            n_min = min(n_i, n_j)

            C[(i, j)] = 2 / n_i + 2 / n_j - 2 + 2 * n_ij / n_max + n_ij / n_min

        nx.set_edge_attributes(self.G, C, "lrc")
        return self.G