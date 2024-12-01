import torch
from torch.utils.data import Dataset

class DBLPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2500):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data.iloc[idx]['input'])
        # print(f"Index: {idx}, Input text: {input_text}...")
        label = self.data.iloc[idx]['d']
        label_tensor = torch.tensor(label, dtype=torch.float32)
        # label = torch.log(label_tensor + 1.01)
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        # print(inputs['input_ids'][0][2000:2500])
        return {'input_ids': inputs['input_ids'].flatten(), 'attention_mask': inputs['attention_mask'].flatten(), 'label': torch.tensor(label)}