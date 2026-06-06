import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class ControlledMPNN(MessagePassing):
    """MPNN with optional per-layer or per-edge message control.

    message_control accepts:
      - scalar: scale every message in every layer
      - shape (L,): scale each propagation layer
      - shape (E,): scale each edge in every layer
      - shape (L, E) or (L, E, D): per-layer edge/message scaling
    """

    def __init__(self, in_fea, out_fea, mpnn_layer, message_scales=None):
        super().__init__()
        self.mpnn_layer = int(mpnn_layer)

        layer_in_dims = [in_fea] + [out_fea] * max(0, self.mpnn_layer - 1)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_fea),
                nn.ReLU(),
                nn.Linear(out_fea, out_fea),
            )
            for in_dim in layer_in_dims
        ])
        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + out_fea, out_fea),
                nn.ReLU(),
                nn.Linear(out_fea, out_fea),
            )
            for in_dim in layer_in_dims
        ])
        self.res_projs = nn.ModuleList([
            nn.Linear(in_dim, out_fea) if in_dim != out_fea else nn.Identity()
            for in_dim in layer_in_dims
        ])
        self.message_scales = message_scales

    def forward(self, x, edge_index, edge_weight, message_control=None, return_messages=False):
        controls = self.message_scales if message_control is None else message_control
        messages = []

        for layer_idx, (msg_mlp, upd_mlp, res_proj) in enumerate(
            zip(self.msg_mlps, self.upd_mlps, self.res_projs)
        ):
            identity = res_proj(x)
            layer_edge_weight = self._controlled_edge_weight(edge_weight, controls, layer_idx)
            m = self.propagate(
                edge_index,
                x=x,
                edge_weight=layer_edge_weight,
                msg_mlp=msg_mlp,
            )
            out = upd_mlp(torch.cat([x, m], dim=-1))
            x = out + identity
            if return_messages:
                messages.append(m)

        if return_messages:
            return x, messages
        return x

    def _controlled_edge_weight(self, edge_weight, controls, layer_idx):
        if controls is None:
            return edge_weight

        control = self._select_layer_control(
            controls,
            layer_idx=layer_idx,
            num_edges=edge_weight.size(0),
            device=edge_weight.device,
            dtype=edge_weight.dtype,
        )
        if control is None:
            return edge_weight
        if control.dim() == 0:
            return edge_weight * control
        if control.dim() == 1:
            return edge_weight * control
        return edge_weight.unsqueeze(-1) * control

    def _select_layer_control(self, controls, layer_idx, num_edges, device, dtype):
        if not torch.is_tensor(controls):
            controls = torch.as_tensor(controls, device=device, dtype=dtype)
        else:
            controls = controls.to(device=device, dtype=dtype)

        if controls.dim() == 0:
            return controls

        if controls.dim() == 1:
            if controls.numel() == self.mpnn_layer:
                return controls[layer_idx]
            if controls.numel() == num_edges:
                return controls
            raise ValueError(
                "1D message_control must have length mpnn_layer "
                f"({self.mpnn_layer}) or num_edges ({num_edges}), got {controls.numel()}."
            )

        if controls.size(0) == self.mpnn_layer:
            layer_control = controls[layer_idx]
            if layer_control.dim() > 0 and layer_control.size(0) != num_edges:
                raise ValueError(
                    "Per-layer message_control must use shape (mpnn_layer, num_edges[, dim]); "
                    f"got {tuple(controls.shape)} for num_edges={num_edges}."
                )
            return layer_control

        if controls.size(0) == num_edges:
            return controls

        raise ValueError(
            "message_control must be scalar, (mpnn_layer,), (num_edges,), "
            f"(mpnn_layer, num_edges[, dim]) or (num_edges, dim); got {tuple(controls.shape)}."
        )

    def message(self, x_j, edge_weight, msg_mlp):
        weight = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
        return msg_mlp(x_j) * weight


MPNN_nk = ControlledMPNN