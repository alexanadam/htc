import torch
from torch import nn
from transformers import AutoModel


class BertModule(nn.Module):
    def __init__(
        self, num_labels, model_name="bert-base-uncased", dropout=0.1, freeze=False
    ):
        super(BertModule, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.bert.base_model.parameters():
                param.requires_grad = False
        # BERT Implementation already contains dropout and classifier layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output = torch.cat([e[:, 0] for e in outputs.hidden_states[-3:]], dim=1) 
        # BERT Implementation already contains dropout and classifier layers
        # output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(output)
        logits = self.classifier(output)

        return logits
