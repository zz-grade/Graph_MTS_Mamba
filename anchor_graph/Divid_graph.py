from typing import Optional, Tuple

import torch






def generate_permutations(n: int, d: int, seed: int = 0, device: Optional[torch.device] = None) -> torch.LongTensor:
    """
    Generate d/2 random permutations π_1..π_{d/2} on {0..n-1}.
    This matches the paper's permutation-based construction (requires even d).

    Returns:
        perms: (K, n) where K = d/2, each row is a permutation of 0..n-1
    """
    assert d % 2 == 0 and d >= 2, "d must be an even integer >= 2"
    K = d // 2
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    perms = []
    base = torch.arange(n, device=device)
    for _ in range(K):
        perm = base[torch.randperm(n, generator=g)]
        perms.append(perm)
    return torch.stack(perms, dim=0)  # (K, n)


def apply_permutations(x: torch.Tensor, perms: torch.LongTensor) -> torch.Tensor:
    """
    Reorder nodes according to each permutation view.

    Args:
        x: (B, T, N, C)
        perms: (K, N) indices in [0..N-1]

    Returns:
        x_views: (B, T, K, N, C)
    """
    B, T, N, C = x.shape
    K, N2 = perms.shape
    assert N == N2

    # Expand perms to (1,1,K,N,1) for gather
    idx = perms.view(1, 1, K, N, 1).expand(B, T, K, N, C)
    x_exp = x.unsqueeze(2).expand(B, T, K, N, C)
    x_views = torch.gather(x_exp, dim=3, index=idx)
    return x_views