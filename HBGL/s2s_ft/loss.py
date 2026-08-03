import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


def get_predecessor_map(hierarchy):
    parent_map = {}
    for node in hierarchy.nodes:
        parents = hierarchy.predecessors(node)
        parent_map[node] = list(parents)

    return parent_map


def get_depths(hierarchy):
    node_depths = np.ones(len(hierarchy))
    roots = [n for n, d in hierarchy.in_degree() if d == 0]

    for root in roots:
        depths = nx.single_source_shortest_path_length(hierarchy, root)
        for node, dist in depths.items():
            node_depths[node] = dist + 1

    return node_depths


def get_children(hierarchy):
    return {n: list(hierarchy.successors(n)) for n in hierarchy.nodes()}



class HierarchicalBCELoss(nn.Module):
    def __init__(
        self,
        hierarchy,
        margin=0.3,
        lambda_MA=1,
        lambda_MD=1,
        lambda_ED=1,
        lambda_ES=1,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_MA = lambda_MA
        self.lambda_MD = lambda_MD
        self.lambda_ED = lambda_ED
        self.lambda_ES = lambda_ES
        self.parent_map = get_predecessor_map(hierarchy)
        self.depths = torch.tensor(get_depths(hierarchy), dtype=torch.float32)
        self.margin = margin
        self.all_siblings = [c for c in get_children(hierarchy).values() if len(c) > 1]
        edges = list(hierarchy.edges())
        self.ed_parents = [p for p, _ in edges]
        self.ed_children = [c for _, c in edges]

    def forward(self, logits, targets):
        device = logits.device

        bce_loss = self.bce(logits, targets)
        logits = logits[:, 1:]
        targets = targets[:, 1:]

        ma_loss = torch.tensor(0.0, device=device)
        md_loss = torch.tensor(0.0, device=device)
        ed_loss = torch.tensor(0.0, device=device)
        es_loss = torch.tensor(0.0, device=device)

        if self.lambda_MA:
            count = 0
            for child_idx, parent_idxs in self.parent_map.items():
                target_child = targets[:, child_idx]
                for parent_idx in parent_idxs:
                    log_p = F.logsigmoid(logits[:, parent_idx])

                    violation = -target_child * log_p
                    ma_loss += violation.mean()
                    count += 1

            if count > 0:
                ma_loss = ma_loss / count

        if self.lambda_MD:
            # md_loss = (depths * targets * F.sigmoid(-logits)).mean()
            md_loss = -(self.depths.to(device) * targets * F.logsigmoid(logits)).mean()

        if self.lambda_ED:
            probs = F.sigmoid(logits)
            p_probs = probs[:, self.ed_parents]
            c_probs = probs[:, self.ed_children]
            
            violations = torch.relu(c_probs - p_probs) ** 2
            
            ed_loss = violations.mean()

        if self.lambda_ES:
            probs = F.sigmoid(logits)
            for siblings in self.all_siblings:
                sib_probs = probs[:, siblings]
                sib_targets = targets[:, siblings]

                true_mask = sib_targets == 1
                false_mask = sib_targets == 0

                valid_mask = true_mask.any(dim=1) & false_mask.any(dim=1)
                if not valid_mask.any():
                    continue

                padded_true = torch.where(
                    true_mask, sib_probs, torch.tensor(2.0, device=logits.device)
                )
                min_true = padded_true.min(dim=1).values

                padded_false = torch.where(
                    false_mask, sib_probs, torch.tensor(-1.0, device=logits.device)
                )
                max_false = padded_false.max(dim=1).values

                violation = torch.relu(max_false - min_true + self.margin)

                violation = violation * valid_mask.float()

                es_loss += violation.sum()

            es_loss = es_loss / (len(self.all_siblings) * logits.size(0))

        total_loss = (
            bce_loss
            + self.lambda_MA * ma_loss
            + self.lambda_MD * md_loss
            + self.lambda_ED * ed_loss
            + self.lambda_ES * es_loss
        )
        return total_loss

    def extra_repr(self):
        return f"lambda_MA: {self.lambda_MA}\n lambda_MD: {self.lambda_MD}\n lambda_ED: {self.lambda_ED}\n lambda_ES: {self.lambda_ES}\n weight: {self.weight is not None}\n soft: {self.soft}\n margin: {self.margin}"
    