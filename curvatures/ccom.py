from collections import defaultdict
import networkx as nx


class CCOM:
    """Approximate Ollivier-Ricci curvature using CCOM.

    This class computes an approximate Ollivier-Ricci curvature for each edge
    based on the CCOM method. The computed curvature values are stored as edge
    attributes named ``"ccom"``.
    """

    def __init__(self, graph: nx.Graph):
        """Initialize the CCOM curvature calculator.

        Args:
            graph (networkx.Graph): The input graph.
        """
        self.G = graph.copy()
        self.neighbors = {node: set(self.G.neighbors(node)) for node in self.G.nodes}
        self.degrees = dict(self.G.degree)

    def calculate_all_edges_ccom(self):
        """Compute CCOM-based approximate Ollivier-Ricci curvature for all edges.

        For each edge, this method first orders the two endpoints by degree,
        enumerates 3-, 4-, and 5-cycle structures passing through the
        edge, and then computes the CCOM-based approximate Ollivier-Ricci
        curvature.

        Returns:
            networkx.Graph: A copy of the input graph with CCOM-based
            approximate Ollivier-Ricci curvature values stored as edge
            attributes named ``"ccom"``.
        """
        # Iterate over all edges in the graph.
        for x, y in self.G.edges:
            # By default, let x be the endpoint with the larger degree,
            # which simplifies the subsequent procedure.
            if self.degrees[y] > self.degrees[x]:
                x, y = y, x

            # Enumerate 3-, 4-, and 5-cycles passing through edge (x, y),
            # and obtain the common-neighbor set as well as the two nested
            # mappings d_out and d_in.
            common, d_out, d_in = self.cycle_enumeration(x, y)

            # Compute the CCOM value for edge (x, y).
            self.curvature_calculation(x, y, common, d_out, d_in)

        return self.G

    def cycle_enumeration(self, x, y):
        """Enumerate 3-, 4-, and 5-cycles passing through edge ``(x, y)``.

        This method enumerates 3-, 4-, and 5-cycle structures passing through
        edge ``(x, y)``. The enumeration results are stored in two nested mappings,
        ``d_out`` and ``d_in``.

        ``d_out`` uses source nodes on the ``x`` side as first-level keys and
        cycle types as second-level keys, and records the target nodes on the
        ``y`` side that each source node can match. ``d_in`` uses target nodes
         on the ``y`` side as first-level keys and cycle types as second-level keys,
         and records the source nodes on the ``x`` side that each target node can match.

        For example, ``d_out[u]["4cyc"]`` records the target nodes that source
        node ``u`` can match through 4-cycle structures, while ``d_in[v]["5cyc"]``
        records the source nodes that target node ``v`` can match through
        5-cycle structures.

        Args:
            x: One endpoint of the edge.
            y: The other endpoint of the edge.

        Returns:
            tuple: A tuple ``(common, d_out, d_in)``, where ``common`` is the set
            of common neighbors of ``x`` and ``y``.
        """
        # Initialize nested mapping structures.
        d_out = defaultdict(lambda: {"4cyc": [], "5cyc": []})
        d_in = defaultdict(lambda: {"3cyc": [], "4cyc": [], "5cyc": []})

        # Obtain the common neighbors of x and y (3-cycle neighbors).
        common = self.neighbors[x] & self.neighbors[y]
        for node in common:
            d_in[node]["3cyc"].append(node)

        # Build candidate neighbor sets of x and y.
        xn_candidates = self.neighbors[x] - {y} - common
        yn_candidates = self.neighbors[y] - {x}

        # Traverse candidate neighbors of x (each as xn).
        for xn in xn_candidates:
            # Traverse candidate neighbors of y (each as yn).
            for yn in yn_candidates:
                # Detect 4-cycles.
                if xn in self.neighbors[yn]:
                    # In this case, x-xn-yn-y-x forms a 4-cycle.
                    d_out[xn]["4cyc"].append(yn)
                    d_in[yn]["4cyc"].append(xn)
                    continue  # No need to continue enumerating 5-cycles.

                # Detect 5-cycles.
                if (self.neighbors[xn] & self.neighbors[yn])- {x, y}:
                    # In this case, there exists a node z such that
                    # x-xn-z-yn-y-x forms a 5-cycle.
                    d_out[xn]["5cyc"].append(yn)
                    d_in[yn]["5cyc"].append(xn)

        return common, d_out, d_in

    # 计算边(x,y)的CCOM
    def curvature_calculation(self, x, y, common, d_out, d_in):
        """Compute the CCOM value for edge ``(x, y)``.

        This method performs greedy probability-mass allocation through
        4-cycle and 5-cycle structures, computes the numbers of effective
        3-, 4-, and 5-cycles, and stores the resulting CCOM-based approximate
        Ollivier-Ricci curvature as the edge attribute ``"ccom"``.

        Args:
            x: One endpoint of the edge.
            y: The other endpoint of the edge.
            common (set): The set of common neighbors of ``x`` and ``y``.
            d_out (defaultdict): A nested mapping that records, for each
                source node on the ``x`` side, the target nodes on the
                ``y`` side that can be matched under 4-, and 5-cycle structures.
            d_in (defaultdict): A nested mapping that records, for each
                target node on the ``y`` side, the source nodes on the
                ``x`` side that can be matched under 3-, 4-, and 5-cycle structures.
        """
        d_x = self.degrees[x]
        d_y = self.degrees[y]
        inv_dx = 1.0 / d_x
        inv_dy = 1.0 / d_y
        num_of_effective_3_cycles = len(common)
        num_of_effective_4_cycles = 0
        num_of_effective_5_cycles = 0

        # Probability mass allocation tables.
        pm_supply = {}  # Remaining probability mass available at source nodes.
        pm_demand = {}  # Remaining probability mass required at target nodes.

        # Keep the probability mass of all common neighbors in place.
        for node, d_in_info in d_in.items():
            if d_in_info["3cyc"]:
                pm_demand[node] = inv_dy - inv_dx
            else:
                pm_demand[node] = inv_dy

        for node in d_out.keys():
            pm_supply[node] = inv_dx

        # Greedy probability-mass allocation for 4-cycles.
        # s_src records the sorted source nodes.
        s_src = sorted(d_out.keys(), key=lambda node: len(d_out[node]["4cyc"]))
        for u in s_src:
            # tgt records the target nodes that u can match through 4-cycles.
            tgt = d_out[u]["4cyc"]

            # s_tgt is the sorted version of tgt.
            s_tgt = sorted(
                tgt,
                key=lambda node: (
                    len(d_in[node]["3cyc"]) +
                    len(d_in[node]["4cyc"]) +
                    len(d_in[node]["5cyc"])
                )
            )

            for v in s_tgt:
                # Probability mass transported from u to v.
                pm = min(pm_supply[u], pm_demand[v])

                # Update variables.
                pm_supply[u] -= pm
                pm_demand[v] -= pm
                num_of_effective_4_cycles += (pm * d_x)

                if pm_supply[u] < 1e-12:
                    break  # The probability mass of u is regarded as exhausted.

        # Greedy probability-mass allocation for 5-cycles.
        s_src = sorted(d_out.keys(), key=lambda node: len(d_out[node]["5cyc"]))
        for u in s_src:
            # tgt records the target nodes that u can match through 5-cycles.
            tgt = d_out[u]["5cyc"]

            if pm_supply[u] < 1e-12:
                continue  # The probability mass of u is regarded as exhausted.

            # s_tgt is the sorted version of tgt.
            s_tgt = sorted(tgt, key=lambda node: len(d_in[node]["5cyc"]))
            for v in s_tgt:
                # Probability mass transported from u to v.
                pm = min(pm_supply[u], pm_demand[v])

                # Update variables.
                pm_supply[u] -= pm
                pm_demand[v] -= pm
                num_of_effective_5_cycles += (pm * d_x)

                if pm_supply[u] < 1e-12:
                    break  # The probability mass of u is regarded as exhausted.

        # Update N_eff,5
        a = 1 - inv_dx - (num_of_effective_3_cycles + num_of_effective_4_cycles + num_of_effective_5_cycles) * inv_dx
        if a < inv_dy:
            num_of_effective_5_cycles -= min((inv_dy - a) * d_x, num_of_effective_5_cycles)

        # Compute parameter t.
        t = sum(pm_demand[node] for node in common)

        # Compute CCOM.
        part1 = -max(
            1 - inv_dx - inv_dy
            - (num_of_effective_3_cycles + num_of_effective_4_cycles + num_of_effective_5_cycles) / d_x
            - t,
            0
        )
        part2 = -max(
            1 - inv_dx - inv_dy
            - (num_of_effective_3_cycles + num_of_effective_4_cycles) / d_x,
            0
        )
        part3 = num_of_effective_3_cycles / d_x
        ccom = part1 + part2 + part3

        self.G.edges[x, y]["ccom"] = ccom
        # self.G.edges[x, y]["num_of_effective_3_cycles"] = num_of_effective_3_cycles
        # self.G.edges[x, y]["num_of_effective_4_cycles"] = num_of_effective_4_cycles
        # self.G.edges[x, y]["num_of_effective_5_cycles"] = num_of_effective_5_cycles



