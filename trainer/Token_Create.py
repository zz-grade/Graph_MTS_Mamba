from random import random

import torch

def precompute_graph_and_tokens(adj, args):
    adj = adj.detach().cpu()
    num_nodes = adj.size(0)

    adj_list = _build_adj_list_from_matrix(adj)
    edge_index = _build_edge_index_from_matrix(adj)

    #deg_list = []
    #for neig in adj_list:
    #    deg_list.append(len(neig))
    #degrees = torch.tensor(deg_list, dtype=torch.long)

    weight_sums = []
    for u in range(num_nodes):
        neighbors = adj_list[u]
        if len(neighbors) == 0:
            weight_sums.append(0)
        else:
            s = adj[u, neighbors].sum().item()
            weight_sums.append(s)
    weight_sums = torch.tensor(weight_sums, dtype=torch.float)
    perm = torch.argsort(weight_sums, descending=False) # 度小在前
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_nodes, dtype=torch.long)

    rng = random.Random(args.seed) if args.seed is not None else random
    tokens_per_node = _sample_tokens_for_all_nodes(num_nodes, adj_list, rng)

    return num_nodes, adj_list, edge_index, tokens_per_node, perm, inv_perm



def _build_adj_list_from_matrix(adj):
    num_nodes = adj.size()
    adj_list = [[] for _ in range(num_nodes)]
    rows, cols = (adj != 0).nonzero(as_tuple=True)
    for u, v in zip(rows.tolist(), cols.tolist()):
        adj_list[u].append(v)
    return adj_list



def _build_edge_index_from_matrix(adj):
    edges = (adj != 0).nonzero(as_tuple=False) # (E, 2)
    edge_index = edges.t().contiguous() # (2, E)
    return edge_index


def _random_walk_one(start, walk_len, adj_list, rng):
    path = [start]
    curr_node = start
    for _ in range(walk_len):
        neighbors = adj_list[curr_node]
        if not neighbors:
            break
        curr_node = rng.choice(neighbors)
        path.append(curr_node)
    return path



def _sample_tokens_for_all_nodes(num_nodes, configs, adj_list, rng):
    """
        为每个节点采样一串子图 token 序列。
        返回：tokens_per_node[v][t] 是节点 v 的第 t 个 token 的节点 id 列表。
        序列长度 L = (m + 1) * s。
    """
    tokens_per_node = [[] for _ in range(num_nodes)]
    for v in range(num_nodes):
        tokens_v = []
        for hop_len in range(configs.max_hop + 1):
            for _ in range(configs.repeat_sample):
                if hop_len == 0:
                    token_nodes = [v]
                else:
                    node_set = set()
                    for _ in range(configs.max_hop):
                        path = _random_walk_one(v, hop_len, adj_list, rng)
                        node_set.update(path)
                    if not node_set:
                        node_set.add(v)
                    token_nodes = list(node_set)
                tokens_v.append(token_nodes)
        tokens_v = list(reversed(tokens_v))
        tokens_per_node[v] = tokens_v
    return tokens_per_node