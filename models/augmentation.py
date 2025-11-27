import torch


def disturbance_correlations(Adj, numremained):
    _, idx = torch.sort(Adj, descending=True, dim=-1)
    new_coe_Adj = torch.zeros_like(Adj)
    topk = idx[:, :, :numremained]
    bat_id = torch.arange(Adj.size(0)).unsqueeze(1).unsqueeze(1)
    sensor_id = torch.arange(Adj.size(1)).unsqueeze(1).unsqueeze(0)
    new_coe_Adj[bat_id, sensor_id, topk] = 1
    rand_coe_Adj = torch.normal(mean=1, std=1, size=Adj.size())
    rand_coe_Adj[bat_id, sensor_id, topk] = 0
    rand_coe_Adj = rand_coe_Adj.cuda() if torch.cuda.is_available() else rand_coe_Adj
    coe_Adj = new_coe_Adj + rand_coe_Adj
    return Adj * coe_Adj

