from torch.utils.data import DataLoader


def get_dataloader(dataset_cls, texts, labels=None, batch=8, shuffle=False, **kwargs):
    data = dataset_cls(texts, labels, **kwargs)

    loader = DataLoader(data, batch_size=batch, shuffle=shuffle)

    return loader
