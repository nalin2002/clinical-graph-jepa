"""(v18) BFS-balanced patch masking for the JEPA pretraining phase.

The partition is adapted from ``clinical_jepa/graph/patches.py`` — the patch
construction behind the patch-based model. It is duplicated rather than
imported: ``tests/test_import_boundaries.py`` forbids ``fawkes`` and
``clinical_jepa`` from importing each other. Only the balanced multi-source BFS
partition is carried over; the coarsened patch graph, positional features and
patch pooling stay in ``clinical_jepa``, because here patches only decide which
nodes are masked — the encoder still works on nodes.

Enabled with ``MASK_STRATEGY=patch`` (``Config.mask_strategy``). Instead of the
i.i.d. Bernoulli node mask (``steps.valid_mask``), each graph is partitioned
into ``jepa_patches`` balanced, connected-ish patches and whole patches are
masked until the ``node_mask`` fraction is reached. The prediction targets are
then coherent local regions rather than scattered nodes, so the JEPA task
cannot be solved from a masked node's immediate neighbors alone.
"""

from __future__ import annotations

from collections import deque

import torch


# ---- balanced multi-source BFS partition (clinical_jepa/graph/patches.py) ----

def _edge_lists(edge_index, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    for s, t in zip(src, dst):
        if 0 <= s < num_nodes and 0 <= t < num_nodes:
            adj[s].append(t)
            adj[t].append(s)
    return adj


def _randperm(n, generator):
    return torch.randperm(n, generator=generator).tolist()


def _bfs_distances(adj, seed):
    dist = [-1] * len(adj)
    dist[seed] = 0
    queue = deque([seed])
    while queue:
        cur = queue.popleft()
        for nb in adj[cur]:
            if dist[nb] < 0:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def _choose_seeds(adj, num_patches, generator):
    """Farthest-point seeds: highest-degree first, then greedily maximize distance."""
    n = len(adj)
    if num_patches >= n:
        return list(range(n))

    tie_rank = {node: i for i, node in enumerate(_randperm(n, generator))}
    degrees = [len(a) for a in adj]
    first = max(range(n), key=lambda u: (degrees[u], -tie_rank[u]))
    seeds = [first]
    best_dist = _bfs_distances(adj, first)
    best_dist = [d if d >= 0 else n + 1 for d in best_dist]

    while len(seeds) < num_patches:
        selected = set(seeds)
        nxt = max(
            (u for u in range(n) if u not in selected),
            key=lambda u: (best_dist[u], degrees[u], -tie_rank[u]),
        )
        seeds.append(nxt)
        dist = _bfs_distances(adj, nxt)
        for i, d in enumerate(dist):
            if d >= 0:
                best_dist[i] = min(best_dist[i], d)
    return seeds


def balanced_bfs_partition(edge_index, num_nodes, num_patches, generator=None):
    """Partition nodes into connected-ish, balanced patches.

    Multi-source BFS keeps patches local. Disconnected components are handled by
    reseeding any unassigned node into the currently smallest patch.
    """
    if num_nodes <= 0:
        return []
    p = max(1, min(num_patches, num_nodes))
    if p == num_nodes:
        return [[i] for i in range(num_nodes)]

    adj = _edge_lists(edge_index, num_nodes)
    seeds = _choose_seeds(adj, p, generator)
    assignment = [-1] * num_nodes
    queues = [deque() for _ in range(p)]
    counts = [0] * p
    target_size = max(1, (num_nodes + p - 1) // p)

    for pid, seed in enumerate(seeds):
        assignment[seed] = pid
        queues[pid].append(seed)
        counts[pid] = 1

    remaining = num_nodes - p
    while remaining > 0:
        progressed = False
        for pid in range(p):
            if counts[pid] >= target_size and any(c < target_size for c in counts):
                continue
            while queues[pid]:
                cur = queues[pid].popleft()
                neighbors = list(adj[cur])
                if neighbors:
                    order = _randperm(len(neighbors), generator)
                    neighbors = [neighbors[i] for i in order]
                for nb in neighbors:
                    if assignment[nb] == -1:
                        assignment[nb] = pid
                        queues[pid].append(nb)
                        counts[pid] += 1
                        remaining -= 1
                        progressed = True
                        break
                if progressed:
                    break
        if not progressed:
            unassigned = [i for i, a in enumerate(assignment) if a == -1]
            if not unassigned:
                break
            node = unassigned[0]
            pid = min(range(p), key=lambda j: counts[j])
            assignment[node] = pid
            queues[pid].append(node)
            counts[pid] += 1
            remaining -= 1

    patches = [[] for _ in range(p)]
    for node, pid in enumerate(assignment):
        patches[pid].append(node)
    return [nodes for nodes in patches if nodes]


# ---- the mask itself ----

def patch_mask(ptr, edge_index, ratio, num_patches, device, generator=None):
    """A node mask of whole BFS-balanced patches, built per graph in the batch.

    Patches are drawn in random order and masked whole until at least ``ratio``
    of the graph's nodes are masked, always leaving the last-drawn patch as
    context. With ``num_patches >= 2`` (enforced by ``Config``) every graph
    keeps >= 1 target and >= 1 context node — the same guarantee
    ``steps.valid_mask`` retries for, here by construction.
    """
    ptr = ptr.detach().cpu()
    edge_index_cpu = edge_index.detach().cpu()
    num_nodes = int(ptr[-1])
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    src, dst = edge_index_cpu[0], edge_index_cpu[1]
    for graph_idx in range(ptr.numel() - 1):
        start, end = int(ptr[graph_idx]), int(ptr[graph_idx + 1])
        n = end - start
        if n < 2:
            raise RuntimeError(f"[FAILURE] graph {graph_idx} has {n} node(s); patch masking needs >= 2.")
        in_graph = (src >= start) & (src < end) & (dst >= start) & (dst < end)
        local_edges = edge_index_cpu[:, in_graph] - start
        patches = balanced_bfs_partition(local_edges, n, num_patches, generator)
        order = _randperm(len(patches), generator)
        wanted = ratio * n
        masked = 0
        for pid in order[:-1]:                       # the last-drawn patch always stays context
            mask[[start + node for node in patches[pid]]] = True
            masked += len(patches[pid])
            if masked >= wanted:
                break
    return mask.to(device)
