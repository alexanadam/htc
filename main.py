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
from data.dataset import TransformerDataset, EmbeddingDataset
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
from training.metrics import plot_loss, plot_metrics, node_weights, plot_importance

if __name__ == "__main__":
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
    logger.info(f"seed {config.SEED}, epochs {config.EPOCHS}")

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
    splits = kfold_splits(input, labels, config.SEED, config.CV_FOLDS)
    weight = None
    if config.WEIGHTED:
        weight = node_weights(labels, hierarchy, config.NORM).to(device)
        plot_importance(
            torch.cat((weight.cpu(), torch.tensor([0.01])), dim=0),
            hierarchy_root,
            classes,
            os.path.join(config.LOG_PATH, "importance.png"),
        )
        logger.info("Weights initialized")

    results = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        # def opt_objective(trial):
        logger.info(f"Starting cv fold {fold}")
        if config.FINETUNE:
            train_input = input.iloc[train_idx]
            val_input = input.iloc[val_idx]
            dataset_cls = TransformerDataset
            dataset_args = {
                "model_name": config.EMB_MODEL,
                "max_length": config.EMB_SIZE,
            }
            model = BertModule(labels.shape[1], config.EMB_MODEL, config.DROPOUT)
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
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.DECAY
        )
        logger.info(f"Optimizer: {optimizer}")

        # lambda_params = {
        #     "lambda_MA": 0, # trial.suggest_float("lambda_ma", 0.1, 2.0),
        #     "lambda_MD": 0, # trial.suggest_float("lambda_md", 0.1, 2.0),
        #     "lambda_ED": 0, # trial.suggest_float("lambda_ed", 0.1, 2.0),
        #     "lambda_ES": trial.suggest_float("lambda_es", 0.1, 2.0),
        #     "margin": trial.suggest_float("margin", 0.1, 0.5)
        # }
        params = {
            "lambda_MA": config.L_MA,
            "lambda_MD": config.L_MD,
            "lambda_ED": config.L_ED,
            "lambda_ES": config.L_ES,
            "margin": config.MARGIN,
        }
        criterion = HierarchicalBCELoss(
            hierarchy, weight, config.SOFT_LOSS, **params
        )
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

        results.append(result)
        #     return result["macro_f1"], result["micro_f1"]

        # study = optuna.create_study(
        #     study_name="bert-lambda-tuning",
        #     directions=["maximize", "maximize"],
        #     sampler=optuna.samplers.TPESampler(seed=config.SEED),
        #     pruner=optuna.pruners.HyperbandPruner()
        # )

        # trials = 10
        # logger.info(f"{trials} trials")
        # study.optimize(opt_objective, n_trials=trials)

        # # logger.info(f"Best Lambdas found: {study.best_params}")
        # for i, trial in enumerate(study.best_trials):
        #     logger.info(f"\nTrade-off Option {i+1}:")
        #     logger.info(f"  macro-F1: {trial.values[0]:.4f}")
        #     logger.info(f"  micro-F1:   {trial.values[1]:.4f}")
        #     logger.info("  Lambdas:")
        #     for key, value in trial.params.items():
        #         logger.info(f"    {key}: {value:.4f}")

    summary = {}
    averaged_losses = {}

    for key in results[0].keys():

        if isinstance(results[0][key], list):
            losses = [fold_res[key] for fold_res in results]

            length = min(len(fold_losses) for fold_losses in losses)
            trunc_losses = [fold_losses[:length] for fold_losses in losses]

            averaged_losses[key] = np.mean(trunc_losses, axis=0).tolist()
        elif isinstance(results[0][key], dict):
            logger.info(f"{key}={[fold_dicts[key] for fold_dicts in results]}")
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
