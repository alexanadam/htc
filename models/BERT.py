from torch import nn
from transformers import AutoModelForSequenceClassification


class BertModule(nn.Module):
    def __init__(self, num_labels, model_name="bert-base-uncased", dropout=0.1):
        super(BertModule, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )
        # BERT Implementation already contains dropout and classifier layers
        # self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # BERT Implementation already contains dropout and classifier layers
        # output = outputs.last_hidden_state[:, 0, :]
        # output = self.dropout(output)
        # logits = self.classifier(output)

        return outputs.logits
