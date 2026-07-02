from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def kfold_splits(data, labels, seed, folds=10):

    mskf = MultilabelStratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )

    return mskf.split(data, labels)
