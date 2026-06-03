import torch
import numpy as np
from training.metrics import ahc_evaluation, standard_evaluation, analysis, violations


class Evaluator:
    def __init__(self, threshold=0.5, average=None):
        self.threshold = threshold
        self.average = average

    def _get_predictions(self, model, dataloader, device):
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    targets = batch.pop("labels").to(device)
                    inputs = {k: v.to(device) for k, v in batch.items()}
                else:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                if isinstance(inputs, dict):
                    logits = model(**inputs)
                else:
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
        self,
        hierarchy,
        classes,
        levels,
        threshold=0.5,
        average=None,
        filename=None,
        npz=None,
    ):
        super().__init__(threshold=threshold, average=average)
        self.hierarchy = hierarchy
        self.classes = classes
        self.levels = levels
        self.file = filename
        self.npz = npz

    def __call__(
        self, model, dataloader, device, analyze=False, save=False
    ):
        y_true, y_pred = self._get_predictions(model, dataloader, device)

        metrics = standard_evaluation(y_true, y_pred, self.average)
        metrics.update(ahc_evaluation(y_true, y_pred, self.hierarchy))
        if analyze and self.file is not None:
            analysis(y_true, y_pred, self.classes, self.file, self.levels)
            metrics.update(violations(y_true, y_pred, self.hierarchy))
            tr = np.sum(y_true, axis=1)
            pr = np.sum(y_pred, axis=1)
            predictions = {
                "tr_mean": np.mean(tr).item(),
                "tr_median": np.median(tr).item(),
                "tr_min": np.min(tr).item(),
                "tr_max": np.max(tr).item(),
                "pr_mean": np.mean(pr).item(),
                "pr_median": np.median(pr).item(),
                "pr_min": np.min(pr).item(),
                "pr_max": np.max(pr).item(),
            }
            metrics.update(predictions)
        # if trainloader is not None:
        #     train_true, train_pred = self._get_predictions(model, trainloader, device)
        #     prob = conditional_probability(train_true, train_pred, self.hierarchy)
        #     error = error_propagation(train_true, train_pred, self.hierarchy, prob)
        if save and self.npz is not None:
            np.savez(self.npz, target=y_true, prediction=y_pred)
        return metrics
