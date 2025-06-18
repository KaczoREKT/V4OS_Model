from torch.utils.data import Dataset
import torch

class WikipediaDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text).ids     # <-- TUTAJ .ids!
            if len(ids) < 2: continue
            for i in range(0, len(ids) - seq_len):
                x = ids[i:i+seq_len]
                y = ids[i+1:i+seq_len+1]
                self.data.append((x, y))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = x + [self.pad_id] * (self.seq_len - len(x))
        y = y + [self.pad_id] * (self.seq_len - len(y))
        return torch.tensor(x), torch.tensor(y)

