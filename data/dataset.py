import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


class TokenDataset(Dataset):
    def __init__(
        self, texts, labels=None, model_name="bert-base-uncased", device="cpu"
    ):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def _compute_embeddings(self, text):
        encoded = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(self.texts, pd.DataFrame):
            row = self.texts.iloc[idx].astype(str).tolist()
            x = f"{self.tokenizer.sep_token}".join(row)
            # x = self._compute_embeddings(x)
        else:
            x = self.texts[idx]
            # x = self._compute_embeddings(self.texts[idx])
        if self.labels is not None:
            y = self.labels[idx]
            # y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x

    def collate_fn(self, batch):
        if self.labels is not None:
            texts = [item[0] for item in batch]
            labels = [item[1] if len(item) > 1 else None for item in batch]

            embeddings = self._compute_embeddings(texts)

            if all(label is not None for label in labels):
                labels = torch.tensor(labels, dtype=torch.long).to(self.device)
                return embeddings, labels
        else:
            embeddings = self._compute_embeddings(batch)
        return embeddings


class TextDataset(Dataset):
    def __init__(
        self, texts, labels=None, model_name="bert-base-uncased", device="cpu"
    ):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def _compute_embeddings(self, text):
        encoded = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state

        return embeddings[:, 0, :]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(self.texts, pd.DataFrame):
            row = self.texts.iloc[idx].astype(str).tolist()
            x = f"{self.tokenizer.sep_token}".join(row)
            # x = self._compute_embeddings(x)
        else:
            x = self.texts[idx]
            # x = self._compute_embeddings(self.texts[idx])
        if self.labels is not None:
            y = self.labels[idx]
            # y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x

    def collate_fn(self, batch):
        if self.labels is not None:
            texts = [item[0] for item in batch]
            labels = [item[1] if len(item) > 1 else None for item in batch]

            embeddings = self._compute_embeddings(texts)

            if all(label is not None for label in labels):
                labels = torch.from_numpy(np.asarray(labels)).to(self.device)
            # if all(len(item) > 1 and item[1] is not None for item in batch):
            #     labels = torch.from_numpy(np.stack([item[1] for item in batch]))

                return embeddings, labels
        else:
            embeddings = self._compute_embeddings(batch)
        return embeddings


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
