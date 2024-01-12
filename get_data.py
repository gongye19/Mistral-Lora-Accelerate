import torch
import pandas as pd
from tqdm import tqdm
import json
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

save_path = './formatted_data'


dataset = load_dataset('gbharti/finance-alpaca')
data = dataset['train']
df = pd.DataFrame(data)

needed_data = df[['instruction', 'output']]
needed_data = needed_data.rename(columns={'instruction':'input', 'output':'target'})
train_df, test_df = train_test_split(needed_data, test_size=0.01)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
# print(train_df)
# print(test_df)
# m=9/0
os.makedirs(save_path,exist_ok=True)
train_df.to_parquet(save_path+'/train.parquet')
test_df.to_parquet(save_path+'/test.parquet')


