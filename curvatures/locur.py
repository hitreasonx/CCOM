import networkx as nx


class LoCur:
    """Approximate Ollivier-Ricci curvature using LoCur.

    This class computes an approximate Ollivier-Ricci curvature for each edge
    based on the LoCur formula. The computed curvature values are stored as
    edge attributes named ``'locur'``.
    """

    def __init__(self, graph: nx.Graph):
        """Initialize the LoCur calculator with a copy of the input graph.

        Args:
            graph (networkx.Graph): The input graph.
        """
        self.G = graph.copy()

    def calculate_all_edges_locur(self):
        """Compute LoCur-based approximate Ollivier-Ricci curvature for every edge.

        For each edge, the number of common neighbors is counted first, which
        corresponds to the number of 3-cycles containing that edge. The LoCur-based
        approximate Ollivier-Ricci curvature is then computed and stored as the
        edge attribute ``'locur'``.

        Returns:
            networkx.Graph: A copy of the input graph with LoCur-based approximate
            Ollivier-Ricci curvature values stored in edge attributes.
        """
        # Precompute the neighbor set of each node for efficient intersection.
        neighbors = {node: set(self.G.neighbors(node)) for node in self.G.nodes}

        # Iterate over all edges and compute the number of common neighbors
        # and the corresponding LoCur approximate curvature.
        for x, y in self.G.edges:
            num_of_3_cycles = len(neighbors[x] & neighbors[y])
            self.G.edges[x, y]['locur'] = self.calculate_locur(
                self.G.degree[x],
                self.G.degree[y],
                num_of_3_cycles
            )
        return self.G

    def calculate_locur(self, d_x, d_y, num_of_3_cycles):
        """Compute the LoCur approximate curvature for a single edge.

        Args:
            d_x (int): Degree of one endpoint of the edge.
            d_y (int): Degree of the other endpoint of the edge.
            num_of_3_cycles (int): Number of 3-cycles containing the edge,
                equivalent to the number of common neighbors of its endpoints.

        Returns:
            float: The LoCur approximate curvature value of the edge.
        """
        part1 = -max(1 - 1 / d_x - 1 / d_y - num_of_3_cycles / min(d_x, d_y), 0)
        part2 = -max(1 - 1 / d_x - 1 / d_y - num_of_3_cycles / max(d_x, d_y), 0)
        part3 = num_of_3_cycles / max(d_x, d_y)
        result = part1 + part2 + part3
        return result