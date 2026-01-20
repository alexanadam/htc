import torch
import numpy as np
from training.metrics import ahc_evaluation, standard_evaluation


class HierarchicalEvaluator:
    def __init__(self, hierarchy, threshold=0.5, average=None):
        self.hierarchy = hierarchy
        self.threshold = threshold
        self.average = average

    def __call__(self, model, dataloader, device):
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)

                logits = model(inputs)

                probs = torch.sigmoid(logits)

                preds = (probs > self.threshold).int().cpu().numpy()
                targets = targets.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        metrics = standard_evaluation(y_true, y_pred, self.average)
        metrics.update(ahc_evaluation(y_true, y_pred, self.hierarchy))
        return metrics

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}: threshold={self.threshold}, average={self.average}"


class Evaluator:
    def __init__(self, threshold=0.5, average=None):
        self.threshold = threshold
        self.average = average

    def __call__(self, model, dataloader, device):
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)

                logits = model(inputs)

                probs = torch.sigmoid(logits)

                preds = (probs > self.threshold).int().cpu().numpy()
                targets = targets.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        metrics = standard_evaluation(y_true, y_pred, self.average)

        return metrics

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}: threshold={self.threshold}, average={self.average}"
