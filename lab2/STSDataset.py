import torch.utils.data
import pandas as pd
from transformers import BertTokenizer

class StsDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(StsDataset, self).__init__()
        self.data = pd.read_csv(file_path, sep='\t', engine='python', on_bad_lines='skip', quoting=3)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        row = self.data.values[key]
        score = float(row[4])
        score = score * 2/5
        score = score - 1
        sent1 = self.tokenizer(row[5], return_tensors='pt')
        sent2 = self.tokenizer(row[6], return_tensors='pt')
        return [sent1, sent2], torch.tensor(score)
