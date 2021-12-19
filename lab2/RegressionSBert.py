import torch
from transformers import BertTokenizer, BertModel


class SBert(torch.nn.Module):

    def __init__(self):
        super(SBert, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, sentences):
        sent1 = sentences[0]
        sent2 = sentences[1]
        left_bert_output = torch.mean((self.bert_layer(input_ids=sent1["input_ids"].squeeze(0))).last_hidden_state, dim=1)
        right_bert_output = torch.mean((self.bert_layer(input_ids=sent2["input_ids"].squeeze(0))).last_hidden_state, dim=1)
        x = self.cosine_similarity(left_bert_output, right_bert_output)
        return x

    def freeze_bert_params(self):
        for param in self.bert_layer.parameters():
            param.requires_grad = False

    def unfreeze_bert_params(self):
        for param in self.bert_layer.parameters():
            param.requires_grad = True
