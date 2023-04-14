from dataset import *
import pandas as pd
from torch.utils.data import DataLoader

dataset = set_b_dataset(path=PROJECT_DIR)
path = pd.read_csv(f'{PROJECT_DIR}dataset/set_b.csv')['fname']
print(path)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(len(train_dataloader))
for data in train_dataloader:
    print(len(data))