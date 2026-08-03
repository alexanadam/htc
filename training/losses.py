import torch
import torch.nn as nn
import torch.nn.functional as F
from data.preprocess import (
    get_predecessor_map,
    get_depths,
    get_children,
)


class HierarchicalBCELoss(nn.Module):
    def __init__(
        self,
        hierarchy,
        weight=None,
        soft=False,
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
        self.weight = weight
        self.soft = soft
        self.margin = margin
        self.all_siblings = [c for c in get_children(hierarchy).values() if len(c) > 1]
        edges = list(hierarchy.edges())
        self.ed_parents = [p for p, _ in edges]
        self.ed_children = [c for _, c in edges]

    def forward(self, logits, targets):
        device = logits.device

        if self.weight is not None:
            self.bce.weight = self.weight.to(device)
        bce_loss = self.bce(logits, targets)

        ma_loss = torch.tensor(0.0, device=device)
        md_loss = torch.tensor(0.0, device=device)
        ed_loss = torch.tensor(0.0, device=device)
        es_loss = torch.tensor(0.0, device=device)

        if self.lambda_MA:
            count = 0
            for child_idx, parent_idxs in self.parent_map.items():
                if self.soft:
                    target_child = F.sigmoid(logits[:, child_idx])
                else:
                    target_child = targets[:, child_idx]
                for parent_idx in parent_idxs:
                    # p_parent = probs[:, parent_idx]
                    log_p = F.logsigmoid(logits[:, parent_idx])

                    # violation = -target_child * torch.log(p_parent)
                    violation = -target_child * log_p
                    if self.weight is not None:
                        violation *= self.weight[parent_idx]
                    ma_loss += violation.mean()
                    count += 1

            if count > 0:
                ma_loss = ma_loss / count

        if self.lambda_MD:
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
        return total_loss, bce_loss, ma_loss, md_loss, ed_loss, es_loss

    def extra_repr(self):
        return f"lambda_MA: {self.lambda_MA}\n lambda_MD: {self.lambda_MD}\n lambda_ED: {self.lambda_ED}\n lambda_ES: {self.lambda_ES}\n weight: {self.weight is not None}\n soft: {self.soft}\n margin: {self.margin}"


class AncestorBCELoss(nn.Module):
    def __init__(self, hierarchy, lambda_reg=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.hierarchy = hierarchy
        self.lambda_reg = lambda_reg
        self.parent_map = get_predecessor_map(self.hierarchy)

    def forward(self, logits, targets):
        device = logits.device

        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        hier_loss = torch.tensor(0.0, device=device)
        count = 0

        for child_idx, parent_idxs in self.parent_map.items():
            # p_child = probs[:, child_idx]
            target_child = targets[:, child_idx]
            for parent_idx in parent_idxs:
                p_parent = probs[:, parent_idx]

                violation = -target_child * torch.log(p_parent)
                hier_loss += violation.mean()
                count += 1

        if count > 0:
            hier_loss = hier_loss / count

        total_loss = bce_loss + self.lambda_reg * hier_loss
        return total_loss, hier_loss

    def extra_repr(self):
        return f"lambda_reg: {self.lambda_reg}"


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        pt = torch.exp(-bce_loss)

        focal_loss = (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

    def extra_repr(self):
        return f"gamma: {self.gamma}"


class HierarchicalFocalLoss(nn.Module):
    def __init__(self, hierarchy, lambda_reg=1, gamma=2):
        super(HierarchicalFocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.hierarchy = hierarchy
        self.lambda_reg = lambda_reg
        self.parent_map = get_predecessor_map(self.hierarchy)

    def forward(self, logits, targets):
        device = logits.device

        bce_loss = self.bce(logits, targets)

        pt = torch.exp(-bce_loss)

        focal_loss = (1 - pt) ** self.gamma * bce_loss

        probs = torch.sigmoid(logits)

        hier_loss = torch.tensor(0.0, device=device)
        # torch.zeros(logits.size(0), device=device)
        count = 0

        for child_idx, parent_idxs in self.parent_map.items():
            target_child = targets[:, child_idx]
            for parent_idx in parent_idxs:
                p_parent = probs[:, parent_idx]

                violation = -target_child * torch.log(p_parent)
                hier_loss += violation.mean()
                count += 1

        if count > 0:
            hier_loss = hier_loss / count

        total_loss = focal_loss + self.lambda_reg * hier_loss
        return total_loss, hier_loss

    def extra_repr(self):
        return f"lambda_reg: {self.lambda_reg}\n gamma: {self.gamma}"


class HybridFocalLoss(nn.Module):
    def __init__(self, hierarchy, lambda_reg=1, gamma=2):
        super(HybridFocalLoss, self).__init__()
        self.hier_bce = HierarchicalBCELoss(hierarchy, lambda_reg)
        self.gamma = gamma
        self.hierarchy = hierarchy
        self.lambda_reg = lambda_reg
        self.parent_map = get_predecessor_map(self.hierarchy)

    def forward(self, logits, targets):
        total_loss, aux_loss = self.hier_bce(logits, targets)

        pt = torch.exp(-total_loss)

        focal_loss = (1 - pt) ** self.gamma * total_loss

        return focal_loss, aux_loss

    def extra_repr(self):
        return f"lambda_reg: {self.lambda_reg}\n gamma: {self.gamma}"
