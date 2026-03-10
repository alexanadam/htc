import torch
import numpy as np
from training.metrics import ahc_evaluation, standard_evaluation, analysis


class Evaluator:
    def __init__(self, threshold=0.5, average=None):
        self.threshold = threshold
        self.average = average

    def _get_predictions(self, model, dataloader, device):
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

        return y_true, y_pred

    def __call__(self, model, dataloader, device):
        y_true, y_pred = self._get_predictions(model, dataloader, device)
        return standard_evaluation(y_true, y_pred, self.average)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}: threshold={self.threshold}, average={self.average}"


class HierarchicalEvaluator(Evaluator):
    def __init__(
        self, hierarchy, classes, levels, threshold=0.5, average=None, filename=None
    ):
        super().__init__(threshold=threshold, average=average)
        self.hierarchy = hierarchy
        self.classes = classes
        self.levels = levels
        self.file = filename

    def __call__(self, model, dataloader, device, analyze=False):
        y_true, y_pred = self._get_predictions(model, dataloader, device)

        metrics = standard_evaluation(y_true, y_pred, self.average)
        metrics.update(ahc_evaluation(y_true, y_pred, self.hierarchy))
        if analyze and self.file is not None:
            analysis(y_true, y_pred, self.classes, self.file, self.levels)
        return metrics
