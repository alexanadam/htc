import torch.nn as nn


class TransformerModule(nn.Module):
    def __init__(self, label_vocab_size, d_model=768, nhead=8, num_layers=6):
        super(TransformerModule, self).__init__()
        self.label_embedding = nn.Embedding(label_vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, label_vocab_size)

    def forward(self, src, tgt):
        tgt_emb = self.label_embedding(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))

        output = self.decoder(tgt_emb, src, tgt_mask=tgt_mask)
        return self.fc_out(output)
