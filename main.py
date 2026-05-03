import argparse
import torch
import numpy as np
import os
import trackio

from configs import config
from data.preprocess import (
    load_FORC,
    load_WOS,
    get_classes,
    get_hierarchy,
    get_levels,
    init_seed,
    plot_importance,
)
from data.embedding import EmbeddingModel
from models.MLP import MLPModule
from utils.logger import setup_logger
from utils.device import get_best_device
from utils.callbacks import EarlyStopper
from data.dataloader import get_dataloader
from data.splits import kfold_splits
from training.evaluator import HierarchicalEvaluator
from training.trainer import Trainer
from training.losses import HierarchicalBCELoss
from training.metrics import plot_loss, plot_metrics, node_importance

if __name__ == "__main__":
    # trackio.show()
    init_seed(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="forc", help="dataset name to use")
    args = parser.parse_args()
    dataset = args.dataset

    if dataset == "forc":
        data_path = config.FORC_PATH
        taxonomy = config.FORC_TAXONOMY
        root = config.FORC_ROOT
        load = load_FORC
    elif dataset == "wos":
        data_path = config.WOS_PATH
        taxonomy = config.WOS_TAXONOMY
        root = config.WOS_ROOT
        load = load_WOS
    elif dataset == "toy":
        data_path = config.TOY_PATH
        taxonomy = config.TOY_TAXONOMY
        root = config.TOY_ROOT
        load = load_FORC
    logger = setup_logger(dataset)

    logger.info(f"Starting {logger.name} run")

    device = get_best_device(logger)

    classes = get_classes(taxonomy, root_excl=root)

    hierarchy_root = get_hierarchy(taxonomy, classes)
    levels = get_levels(hierarchy_root)
    hierarchy = get_hierarchy(taxonomy, classes, root_excl=root)

    data, labels = load(data_path, classes)
    emb_model = EmbeddingModel(config.EMB_MODEL, logger, device=device)
    embeddings = emb_model.compute_embeddings(data, config.EMB_BATCH, config.EMB_SIZE)
    splits = kfold_splits(embeddings, labels)
    if config.WEIGHTED:
        weight = node_importance(labels, hierarchy).to(device)
    else:
        weight = None

    plot_importance(
        torch.cat((weight.cpu(), torch.tensor([0.01])), dim=0),
        hierarchy_root,
        classes,
        os.path.join(config.LOG_PATH, "importance.png"),
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        logger.info(f"Starting cv fold {fold}")

        train_loader = get_dataloader(
            embeddings[train_idx],
            labels[train_idx],
            batch=config.BATCH_SIZE,
            shuffle=True,
        )
        val_loader = get_dataloader(
            embeddings[val_idx],
            labels[val_idx],
            batch=config.BATCH_SIZE,
            shuffle=False,
        )

        input_dim = next(iter(train_loader))[0].size(1)

        model = MLPModule(input_dim, config.HIDDEN_DIM, labels.shape[1], config.DROPOUT)
        logger.info(f"Model: {model}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        logger.info(f"Optimizer: {optimizer}")
        # criterion = HybridFocalLoss(hierarchy, config.LAMBDA, config.GAMMA)
        # criterion = HierarchicalFocalLoss(hierarchy, config.LAMBDA, config.GAMMA)
        criterion = HierarchicalBCELoss(hierarchy, config.LAMBDA, weight)
        # criterion = torch.nn.BCEWithLogitsLoss(weight=weight)
        logger.info(f"Criterion: {criterion}")

        # evaluator = Evaluator(average=config.AVERAGES)
        evaluator = HierarchicalEvaluator(
            hierarchy,
            classes,
            levels,
            average=config.AVERAGES,
            filename=config.LOG_ANALYSIS,
            npz=config.NPZ,
        )
        logger.info(f"Evaluator: {evaluator}")
        early_stopper = EarlyStopper(patience=config.PATIENCE)
        trainer = Trainer(
            model,
            optimizer,
            criterion,
            device,
            logger,
            train_loader,
            val_loader=val_loader,
            evaluator=evaluator,
            early_stopper=early_stopper,
            # checkpoint_dir=config.CHECKPOINT_PATH,
        )
        trackio.init(
            project=logger.name,
            group=f"fold{fold}",
            config={
                "dataset": dataset,
                "epochs": config.EPOCHS,
                "learning_rate": config.LEARNING_RATE,
                "patience": config.PATIENCE,
                "batch_size": config.BATCH_SIZE,
                "model": model.__repr__(),
                "optimizer": optimizer.__repr__(),
                "criterion": criterion.__repr__(),
                "weighted": config.WEIGHTED,
                "evaluator": evaluator.__repr__(),
            },
            auto_log_gpu=True,
        )

        result = trainer.train(config.EPOCHS)
        plot_loss(
            {key: value for key, value in result.items() if isinstance(value, list)},
            os.path.join(config.LOG_PATH, f"loss_{fold}.png"),
            logger,
        )
        results.append(result)

    summary = {}
    averaged_losses = {}

    for key in results[0].keys():

        if isinstance(results[0][key], list):
            losses = [fold_res[key] for fold_res in results]

            length = min(len(fold_losses) for fold_losses in losses)
            trunc_losses = [fold_losses[:length] for fold_losses in losses]

            averaged_losses[key] = np.mean(trunc_losses, axis=0).tolist()

        else:
            metrics = [fold_metrics[key] for fold_metrics in results]
            summary[key] = {
                "mean": np.mean(metrics),
                "std": np.std(metrics),
            }

    for metric, stats in summary.items():
        logger.info(f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    plot_loss(averaged_losses, os.path.join(config.LOG_PATH, "avg_loss.png"), logger)
    plot_metrics(summary, os.path.join(config.LOG_PATH, "metrics.png"), logger)

    trackio.finish()
