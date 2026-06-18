from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import copy
import time
from curvatures import *


# Record the attribute name corresponding to each curvature type
curvature_attr_names = {
    'LRC': 'lrc',
    'FRC': 'frc',
    'BFC': 'bfc',
    'LoCur': 'locur',
    'ALU': 'alu',
    'MWPM': 'mwpm',
    'CCOM': 'ccom',
}


class CurvatureBasedCD:
    """Community detection based on edge curvature."""

    def __init__(self,graph, curvature_name, number_of_communities=None, minimum_community_size=None, chunk_size=1):
        """Initialize the curvature-based community detector."""
        # Working graph (used for curvature computation and edge removal)
        self.G = copy.deepcopy(graph)
        # Number of communities
        self.number_of_communities = number_of_communities
        # Minimum community size
        self.minimum_community_size = minimum_community_size
        # Number of negative-curvature edges removed in each iteration
        self.chunk_size = chunk_size
        # Curvature name and its corresponding edge attribute
        self.curvature_name = curvature_name
        self.attr_name = curvature_attr_names.get(curvature_name)

        # Compute curvature on the full graph once at initialization
        self._recompute_curvature()
        # Original graph
        self.G_original = copy.deepcopy(self.G)

    def _recompute_curvature(self):
        """Recompute curvature on the current graph."""
        if self.curvature_name == 'LRC':
            self.G = compute_lrc(self.G)
        elif self.curvature_name == 'FRC':
            self.G = compute_frc(self.G)
        elif self.curvature_name == 'BFC':
            self.G = compute_bfc(self.G)
        elif self.curvature_name == 'LoCur':
            self.G = compute_locur(self.G)
        elif self.curvature_name == 'ALU':
            self.G = compute_alu(self.G)
        elif self.curvature_name == 'MWPM':
            self.G = compute_mwpm(self.G)
        elif self.curvature_name == 'CCOM':
            self.G = compute_ccom(self.G)
        else:
            raise ValueError(f"Unknown curvature name: {self.curvature_name}")

    def _get_edge_curvature(self, u, v):
        """Return the curvature of edge (u, v)."""
        return self.G[u][v].get(self.attr_name)

    def remove_edges_with_negative_curvature(self):
        """Iteratively remove edges with negative curvature."""
        num_of_removed_edges = 0
        last_time = time.time()
        iteration = 0

        while self.G.number_of_edges() > 0:
            iteration += 1

            # 1. Retrieve curvature values for all edges
            edges_curvature = []
            for u, v in self.G.edges():
                c = self._get_edge_curvature(u, v)
                edges_curvature.append((u, v, c))

            # 2. Sort edges by curvature in ascending order
            edges_curvature.sort(key=lambda x: x[2])
            min_c = edges_curvature[0][2]

            # 3. Stopping condition: stop partitioning if the minimum curvature is already non-negative (>= 0)
            if min_c >= 0:
                break

            # 4. Remove up to chunk_size edges with the smallest negative curvature
            edges_removed = set()
            for i in range(min(self.chunk_size, len(edges_curvature))):
                u, v, c = edges_curvature[i]
                if c < 0:
                    self.G.remove_edge(u, v)
                    num_of_removed_edges += 1
                    edges_removed.add((u, v))
                else:
                    break

            if time.time() - last_time > 60:
                print(f"{num_of_removed_edges} edges have been removed so far")
                last_time = time.time()

            # 5. Recompute curvature on the entire graph
            self._recompute_curvature()

        # All negative-curvature edges have been removed
        print(f"Number of removed edges : {num_of_removed_edges}")

    def _preferential_attachment(self, communities):
        """Merge small fragmented communities by preferential attachment."""
        # Convert to a list of sets for easier manipulation
        communities = [set(c) for c in communities]
        last_time = time.time()
        num_of_processed_communities = 0

        # Used to store isolated communities
        isolated_communities = []

        while True:
            # Sort communities by size
            communities.sort(key=len)

            # Check whether merging is still needed
            # Stopping condition 1: neither the minimum community size nor the number of communities is provided
            if self.number_of_communities is None and self.minimum_community_size is None:
                break

            # Stopping condition 2: the smallest community already satisfies the minimum size requirement
            smallest = communities[0]
            if self.minimum_community_size is not None and len(smallest) >= self.minimum_community_size:
                break

            # Stopping condition 3: the current number of communities is less than or equal to the target number
            if (self.number_of_communities is not None and
                    (len(communities) + len(isolated_communities)) <= self.number_of_communities):
                break

            # Start merging logic: find the best target community for the smallest one
            best_target_idx = -1
            max_connections = -1

            # Traverse all other communities
            for i in range(1, len(communities)):
                target = communities[i]

                # Count the number of edges connecting smallest and target in the original graph
                connections = 0

                for u in smallest:
                    for v in target:
                        if self.G_original.has_edge(u, v):
                            connections += 1

                if connections > max_connections:
                    max_connections = connections
                    best_target_idx = i

            # Perform merging
            if best_target_idx != -1 and max_connections > 0:
                target_comm = communities[best_target_idx]
                target_comm.update(smallest)
                communities.pop(0)  # Remove the fragmented community that has been merged
            else:
                # The smallest community has no edge connection to any remaining community in the original graph
                print(f"Found an isolated community (Size: {len(smallest)})")
                # Strategy: treat it as an independent small community
                isolated_communities.append(smallest)
                communities.pop(0)

            num_of_processed_communities += 1
            if time.time() - last_time > 60:
                print(f"Number of processed fragmented communities so far: {num_of_processed_communities}")
                last_time = time.time()

        return communities + isolated_communities

    def detect(self):
        """Run the full community detection pipeline."""
        start_time = time.time()
        print(f"Starting Community Detection using {self.curvature_name}...")

        # Phase 1: Calculate curvature and remove negative-curvature edges
        self.remove_edges_with_negative_curvature()

        # Get the current connected components
        initial_communities = list(nx.connected_components(self.G))
        print(f"All negative-curvature edges removed. Found {len(initial_communities)} components.")

        # Phase 2: Preferential attachment
        final_communities = self._preferential_attachment(initial_communities)

        # Convert the result to a list of lists
        final_communities = [list(c) for c in final_communities]

        end_time = time.time()
        print(f"Detection finished in {end_time - start_time:.2f}s.")
        print(f"Final communities count: {len(final_communities)}")

        return final_communities