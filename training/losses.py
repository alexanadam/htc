import torch
import torch.nn as nn
from data.preprocess import get_parents_list


class HierarchicalBCELoss(nn.Module):
    def __init__(self, hierarchy, lambda_reg=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.hierarchy = hierarchy
        self.lambda_reg = lambda_reg
        self.parent_map = get_parents_list(self.hierarchy)

    def forward(self, logits, targets):
        device = logits.device

        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        hier_loss = torch.tensor(0.0, device=device)
        count = 0

        for child_idx, parent_idxs in self.parent_map.items():
            p_child = probs[:, child_idx]
            for parent_idx in parent_idxs:
                p_parent = probs[:, parent_idx]

                violation = p_child * (1.0 - p_parent)
                hier_loss += violation.mean()
                count += 1

        if count > 0:
            hier_loss = hier_loss / count

        total_loss = bce_loss + self.lambda_reg * hier_loss
        return total_loss, hier_loss


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
