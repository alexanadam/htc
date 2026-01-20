import torch

from configs import config
from data.preprocess import load_WOS, get_classes, get_hierarchy
from models.MLP import MLPModule
from utils.logger import setup_logger
from utils.device import get_best_device
from data.dataloader import get_dataloader
from data.splits import kfold_splits
from training.evaluator import HierarchicalEvaluator
from training.trainer import Trainer

from training.losses import HierarchicalBCELoss

if __name__ == "__main__":
    data_path = config.WOS_PATH
    taxonomy = config.WOS_TAXONOMY

    logger = setup_logger(config.LOG_NAME)
    logger.info(f"Starting {logger.name} for {data_path} and {taxonomy}")
    device = get_best_device(logger)

    classes = get_classes(taxonomy, root_excl="root")
    # hierarchy = get_hierarchy(config.FORC_TAXONOMY, classes, root="NLP")
    hierarchy_root = get_hierarchy(taxonomy, classes)
    hierarchy = get_hierarchy(taxonomy, classes, root_excl="root")

    X, Y = load_WOS(data_path, classes)
    splits = kfold_splits(X, Y)

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Starting cv fold {fold+1}")

        train_loader = get_dataloader(
            X.iloc[train_idx],
            Y[train_idx],
            batch=config.BATCH_SIZE,
            shuffle=True,
            device=device,
        )
        val_loader = get_dataloader(
            X.iloc[val_idx],
            Y[val_idx],
            batch=config.BATCH_SIZE,
            shuffle=False,
            device=device,
        )

        input_dim = next(iter(train_loader))[0].size(1)

        model = MLPModule(input_dim, config.HIDDEN_DIM, Y.shape[1], config.DROPOUT)
        logger.info(f"Model: {model}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        logger.info(f"Optimizer: {optimizer}")
        criterion = HierarchicalBCELoss(hierarchy, lambda_reg=0.5)
        # criterion = torch.nn.BCEWithLogitsLoss()
        logger.info(f"Criterion: {criterion}")

        # evaluator = Evaluator(average=config.AVERAGES)
        evaluator = HierarchicalEvaluator(hierarchy_root, average=config.AVERAGES)
        logger.info(f"Evaluator: {evaluator}")
        trainer = Trainer(
            model,
            optimizer,
            criterion,
            device,
            logger,
            train_loader,
            val_loader,
            evaluator,
            # checkpoint_dir=config.CHECKPOINT_PATH,
        )
        trainer.train(config.EPOCHS)
