from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np


class MWPM:
    """Approximate Ollivier-Ricci curvature using MWPM.

    This class computes an approximate Ollivier-Ricci curvature for each edge
    based on a minimum-weight perfect matching formulation. The computed
    curvature values are stored as edge attributes named ``"mwpm"``.
    """

    def __init__(self, G: nx.Graph):
        """Initialize the MWPM curvature calculator.

        Args:
            G (networkx.Graph): The input graph.
        """
        self.G = G.copy()
        self.neighbors = {node: set(self.G.neighbors(node)) for node in self.G.nodes}

    def _distance_between_neighbors(self, u, v):
        """Compute the shortest-path distance between two nodes.

        The distance is evaluated using local neighborhood information:
        identical nodes have distance 0, adjacent nodes have distance 1,
        nodes with common neighbors have distance 2, and all remaining cases
        are assigned distance 3.

        Args:
            u: The first node.
            v: The second node.

        Returns:
            int: The shortest-path distance between ``u`` and ``v``.
        """

        # Case 1: The two nodes are identical.
        if u == v:
            return 0

        # Case 2: The two nodes are directly adjacent.
        if v in self.neighbors[u]:
            return 1

        # Case 3: The two nodes share at least one common neighbor.
        if self.neighbors[u] & self.neighbors[v]:
            return 2

        # Case 4: The remaining cases are treated as distance 3.
        return 3

    def _build_bipartite_matrix(self, x, y):
        """Build the bipartite cost matrix for the edge ``(x, y)``.

        The endpoint with the smaller degree is used as the splitting side.
        Its neighbors are expanded to match the degree of the other endpoint,
        producing a square cost matrix for the minimum-weight perfect matching.

        Args:
            x: One endpoint of the edge.
            y: The other endpoint of the edge.

        Returns:
            tuple[np.ndarray, int]: The cost matrix and its size.
        """
        # Ensure that x is the endpoint with the smaller degree, so that
        # the lower-degree side is used for node splitting.
        if self.G.degree(x) > self.G.degree(y):
            x, y = y, x

        deg_x = self.G.degree(x)
        deg_y = self.G.degree(y)

        k = deg_y
        cost_matrix = np.zeros((k, k), dtype=int)  # Cost/distance matrix.

        # Splitting factor a and remainder b.
        a = deg_y // deg_x
        b = deg_y % deg_x

        neighbors_x = list(self.neighbors[x])
        neighbors_y = list(self.neighbors[y])

        # Expand neighbors and fill the cost matrix.
        row_idu = 0
        for u in neighbors_x:
            # Compute distances from the current node u to all nodes in N(y).
            for col_idu, v in enumerate(neighbors_y):
                dist = self._distance_between_neighbors(u, v)
                cost_matrix[row_idu, col_idu] = dist

            # Copy the current row a - 1 times to simulate node splitting.
            if a > 1:
                cost_matrix[row_idu + 1: row_idu + a, :] = cost_matrix[row_idu, :]

            row_idu += a

        # If deg_y is not divisible by deg_x, fill the remaining rows with 3.
        if b != 0:
            cost_matrix[-b:, :] = 3

        return cost_matrix, k

    def _emd_via_mwpm(self, cost_matrix, k):
        """Compute EMD using minimum-weight perfect matching.

        Args:
            cost_matrix (np.ndarray): The cost matrix for bipartite matching.
            k (int): The size of the square cost matrix.

        Returns:
            float: The Earth Mover's Distance value.
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        emd_val = total_cost / k
        return emd_val

    def calculate_all_edges_curvature(self):
        """Compute MWPM-based approximate curvature for every edge in the graph.

        For each edge, this method builds the corresponding bipartite cost
        matrix, computes the EMD via minimum-weight perfect matching, and then
        stores the curvature value as the edge attribute ``"mwpm"``.

        Returns:
            networkx.Graph: A copy of the input graph with MWPM approximate curvature
            values stored in edge attributes.
        """

        for x, y in self.G.edges:
            # Build the cost/distance matrix for the current edge (x, y).
            cost_matrix, k = self._build_bipartite_matrix(x, y)

            # Compute EMD and the corresponding curvature.
            emd_val = self._emd_via_mwpm(cost_matrix, k)
            curvature = 1 - emd_val

            # Store the curvature value as an edge attribute.
            self.G.edges[x, y]["mwpm"] = curvature

        return self.G