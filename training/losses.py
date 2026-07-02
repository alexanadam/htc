import torch
import torch.nn as nn
import torch.nn.functional as F
from data.preprocess import get_parents_list, get_predecessor_map


class HierarchicalBCELoss(nn.Module):
    def __init__(self, hierarchy, lambda_reg=1, weight=None, soft=False):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_reg = lambda_reg
        self.parent_map = get_parents_list(hierarchy)
        self.weight = weight
        self.soft = soft

    def forward(self, logits, targets):
        device = logits.device

        if self.weight is not None:
            self.bce.weight = self.weight.to(device)
        bce_loss = self.bce(logits, targets)
        # probs = torch.sigmoid(logits)

        hier_loss = torch.tensor(0.0, device=device)
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
                hier_loss += violation.mean()
                count += 1

        if count > 0:
            hier_loss = hier_loss / count

        total_loss = bce_loss + self.lambda_reg * hier_loss
        return total_loss, hier_loss

    def extra_repr(self):
        return f"lambda_reg: {self.lambda_reg}\n weight: {self.weight is not None}\n soft: {self.soft}"


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
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
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


# class DistanceBCELoss(nn.Module):
#     def __init__(self, hierarchy, distances, lambda_reg):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.hierarchy = hierarchy
#         self.distances = distances
#         self.lambda_reg = lambda_reg

#     def forward(self, logits, targets):
#         device = logits.device

#         bce_loss = self.bce(logits, targets)
#         probs = torch.sigmoid(logits)

#         dist_loss = torch.tensor(0.0, device=device)
#         count = 0
#         probs

#         if count > 0:
#             dist_loss = dist_loss / count

#         total_loss = bce_loss + self.lambda_reg * dist_loss
#         return total_loss, dist_loss

#     def extra_repr(self):
#         return f"lambda_reg: {self.lambda_reg}"


# class AverageHierarchicalLoss(nn.Module):
#     def __init__(self, hierarchy):
#         self.bce = nn.BCEWithLogitsLoss()
#         self.hierarchy = hierarchy

#     def forward(self, logits, targets):
#         device = logits.device

#         bce_loss = self.bce(logits, targets)
#         probs = torch.sigmoid(logits)

#         hier_loss = torch.tensor(0.0, device=device)
#         count = 0

#         ahc = []
#         graph = self.hierarchy.to_undirected()
#         for prediction, true in zip(probs, targets):
#             distances = []
#             pred_indices = np.where(prediction == 1)[0]
#             true_indices = np.where(true == 1)[0]
#             for pred in pred_indices:
#                 min_d = np.inf
#                 for label in true_indices:
#                     d = nx.shortest_path_length(graph, pred, label)
#                     min_d = min(min_d, d)
#                 distances.append(min_d)
#             ahc.append(np.mean(distances))

#         total_loss = bce_loss + hier_loss
#         return total_loss
