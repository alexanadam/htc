import argparse
import torch
import numpy as np
import os
import trackio

from configs import config
from data.preprocess import (
    load_FORC,
    load_WOS,
    load_saved,
    get_classes,
    get_hierarchy,
    get_levels,
    init_seed,
)
from data.embedding import EmbeddingModel
from data.dataset import BERTDataset, EmbeddingDataset
from models.MLP import MLPModule
from models.BERT import BertModule
from utils.logger import setup_logger
from utils.device import get_best_device
from utils.callbacks import EarlyStopper
from data.dataloader import get_dataloader
from data.splits import kfold_splits
from training.evaluator import HierarchicalEvaluator
from training.trainer import Trainer

from training.losses import HierarchicalBCELoss
from training.metrics import plot_loss, plot_metrics, node_importance, plot_importance

if __name__ == "__main__":
    # trackio.show()
    init_seed(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="forc",
        choices=["forc", "wos", "toy"],
        help="dataset name to use",
    )
    args = parser.parse_args()

    dataset = {
        "forc": (config.FORC_PATH, config.FORC_TAXONOMY, config.FORC_ROOT, load_FORC),
        "saved": (config.FORC_PATH, config.FORC_TAXONOMY, config.FORC_ROOT, load_saved),
        "wos": (config.WOS_PATH, config.WOS_TAXONOMY, config.WOS_ROOT, load_WOS),
        "toy": (config.TOY_PATH, config.TOY_TAXONOMY, config.TOY_ROOT, load_FORC),
    }
    if args.dataset not in dataset:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

    data_path, taxonomy, root, load_fn = dataset[args.dataset]

    logger = setup_logger(args.dataset)

    logger.info(f"Starting {logger.name} run")

    device = get_best_device(logger)

    classes = get_classes(taxonomy, root_excl=root)

    hierarchy_root = get_hierarchy(taxonomy, classes)
    levels = get_levels(hierarchy_root)
    hierarchy = get_hierarchy(taxonomy, classes, root_excl=root)

    data, labels = load_fn(data_path, classes)
    if config.FINETUNE or args.dataset == "saved":
        input = data
    else:
        emb_model = EmbeddingModel(config.EMB_MODEL, logger, device=device)
        input = emb_model.compute_embeddings(data, config.EMB_BATCH, config.EMB_SIZE)
    splits = kfold_splits(input, labels)
    weight = None
    if config.WEIGHTED:
        weight = node_importance(labels, hierarchy, config.NORM).to(device)
        plot_importance(
            torch.cat((weight.cpu(), torch.tensor([0.01])), dim=0),
            hierarchy_root,
            classes,
            os.path.join(config.LOG_PATH, "importance.png"),
        )
        logger.info("Weights initialized")

    results = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        logger.info(f"Starting cv fold {fold}")
        if config.FINETUNE:
            train_input = input.iloc[train_idx]
            val_input = input.iloc[val_idx]
            dataset_cls = BERTDataset
            dataset_args = {
                "model_name": config.EMB_MODEL,
                "max_length": config.EMB_SIZE,
            }
            model = BertModule(labels.shape[1], config.EMB_MODEL, config.DROPOUT)
            for param in model.bert.parameters():
                param.requires_grad = False

            for param in model.bert.classifier.parameters():
                param.requires_grad = True
        else:
            train_input = input[train_idx]
            val_input = input[val_idx]
            dataset_cls = EmbeddingDataset
            dataset_args = {}
            model = MLPModule(
                input.shape[1], config.HIDDEN_DIM, labels.shape[1], config.DROPOUT
            )

        train_loader = get_dataloader(
            dataset_cls,
            train_input,
            labels[train_idx],
            batch=config.BATCH_SIZE,
            shuffle=True,
            **dataset_args,
        )
        val_loader = get_dataloader(
            dataset_cls,
            val_input,
            labels[val_idx],
            batch=config.BATCH_SIZE,
            shuffle=False,
            **dataset_args,
        )

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
        if config.TRACK:
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

        result = trainer.train(config.EPOCHS, config.TRACK)
        plot_loss(
            {key: value for key, value in result.items() if isinstance(value, list)},
            os.path.join(config.LOG_PATH, f"loss_{fold}.png"),
            logger,
        )

        # indices = result["indices"]
        # rest = np.setdiff1d(np.arange(len(val_idx)), indices)
        # if len(rest) >= len(indices):
        #     selected = rest[torch.randperm(len(rest))[: len(indices)]]
        # else:
        #     selected = rest
        # final_idx = np.concat([indices, selected])
        # sliced_data.append(input[val_idx][final_idx])
        # sliced_labels.append(labels[val_idx][final_idx])

        # result.pop("indices")
        results.append(result)

    # sliced_data = torch.cat(sliced_data, dim=0)
    # sliced_labels = np.concat(sliced_labels)

    # torch.save(
    #     {"data": sliced_data, "labels": sliced_labels},
    #     os.path.join(config.FORC_PATH, "sliced.pt"),
    # )

    summary = {}
    averaged_losses = {}
    # sliced = []

    for key in results[0].keys():

        if isinstance(results[0][key], list):
            losses = [fold_res[key] for fold_res in results]

            length = min(len(fold_losses) for fold_losses in losses)
            trunc_losses = [fold_losses[:length] for fold_losses in losses]

            averaged_losses[key] = np.mean(trunc_losses, axis=0).tolist()
        # elif isinstance(results[0][key], np.ndarray):
        #     for fold, (train_idx, val_idx) in enumerate(splits):
        #         indices = results[fold][key]
        #         rest = np.setdiff1d(np.arange(len(val_idx)), indices)
        #         if len(rest) >= len(indices):
        #             selected = rest[torch.randperm(len(rest))[: len(indices)]]
        #         else:
        #             selected = rest
        #         final_idx = torch.cat([indices, selected])
        #         sliced.append(input[val_idx][final_idx])

        #     sliced = torch.cat(sliced, dim=0)
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

    if config.TRACK:
        trackio.finish()
