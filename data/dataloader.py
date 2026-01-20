from torch.utils.data import DataLoader
from data.dataset import TextDataset


def get_dataloader(texts, labels=None, batch=8, shuffle=False, device="cpu"):
    if labels is not None:
        data = TextDataset(texts, labels, device=device)
    else:
        data = TextDataset(texts, device=device)

    fn = getattr(data, "collate_fn", None)

    loader = DataLoader(data, batch_size=batch, shuffle=shuffle, collate_fn=fn)

    return loader
