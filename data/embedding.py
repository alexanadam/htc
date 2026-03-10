import torch
from transformers import BertTokenizer, BertModel


class EmbeddingModel:
    def __init__(self, model_name, logger, device="cpu"):
        self.device = device
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def compute_embeddings(self, texts, batch_size=8, max_length=512):
        self.logger.info(f"Computing embeddings for {len(texts)} texts...")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch = (
                batch.astype(str).agg(self.tokenizer.sep_token.join, axis=1).tolist()
            )

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu())

            self.logger.info(
                f"Batch {i//batch_size+1}/{len(texts)//batch_size+1} completed"
            )

        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()

        return torch.cat(all_embeddings)


class TokenEmbeddingModel:
    def __init__(self, model_name, logger, device="cpu"):
        self.device = device
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def compute_embeddings(self, texts, batch_size=8, max_length=512):
        self.logger.info(f"Computing token embeddings for {len(texts)} texts...")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch = (
                batch.astype(str).agg(self.tokenizer.sep_token.join, axis=1).tolist()
            )

            encoded = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            embeddings = outputs.last_hidden_state

            all_embeddings.append(embeddings.cpu())

            self.logger.info(
                f"Batch {i//batch_size+1}/{len(texts)//batch_size+1} completed"
            )

        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()

        return torch.cat(all_embeddings)
