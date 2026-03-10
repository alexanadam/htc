from torch.utils.data import DataLoader
from data.dataset import EmbeddingDataset


def get_dataloader(texts, labels=None, batch=8, shuffle=False):
    data = EmbeddingDataset(texts, labels)

    loader = DataLoader(data, batch_size=batch, shuffle=shuffle)

    return loader
