import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, AutoTokenizer


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


class BERTDataset(Dataset):
    def __init__(
        self, texts, labels=None, model_name="bert-base-uncased", max_length=512
    ):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(self.texts, pd.DataFrame):
            row = self.texts.iloc[idx].astype(str).tolist()
            x = f"{self.tokenizer.sep_token}".join(row)
        else:
            x = self.texts[idx]
        y = self.labels[idx]

        encoded = self.tokenizer(
            x,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "labels": torch.tensor(y, dtype=torch.float),
        }


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class TransformerDataset(Dataset):
    def __init__(
        self, texts, labels=None, model_name="bert-base-uncased", max_length=512
    ):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(self.texts, pd.DataFrame):
            row = self.texts.iloc[idx].astype(str).tolist()
            x = f"{self.tokenizer.sep_token}".join(row)
        else:
            x = self.texts[idx]
        y = self.labels[idx]

        return x, y

    def collate_fn(self, batch):
        texts = [item[0] for item in batch]
        labels = [item[1] if len(item) > 1 else None for item in batch]
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels),
        }
