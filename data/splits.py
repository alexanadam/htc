import configs.config as config

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def kfold_splits(data, labels, folds=None):

    if folds is None:
        folds = config.CV_FOLDS

    mskf = MultilabelStratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config.SEED
    )

    return mskf.split(data, labels)
