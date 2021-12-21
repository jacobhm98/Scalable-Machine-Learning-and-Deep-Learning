import torch.utils.data
from transformers import BertTokenizer
import datasets


class SNLIDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(SNLIDataset, self).__init__()
        self.dataset = datasets.load_dataset('snli', split='train')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.dataset.dataset_size

    def __getitem__(self, key):
        premise = self.dataset.data['premise'][key]
        hypothesis = self.dataset.data['premise'][key]
        label = self.dataset.data['label'][key]
        sent1 = self.tokenizer(str(premise), return_tensors='pt')
        sent2 = self.tokenizer(str(hypothesis), return_tensors='pt')
        return [sent1, sent2], torch.tensor(label.as_py())
