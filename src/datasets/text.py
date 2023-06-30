import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torch import nn
from base import base_dataset
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

"""
# 从 sentence embedding. 2. 对比 k-means 效果 3. 看一看sentiment analyze效果
# 3. 目的：没有label并且任务复杂继续数据清洗（entertaining, jokes）
# 4. todo... text style detection?
# 5. 区分是不是一个人所写的内容：plagirithm detection
# 6. 还有什么别的方向？

"""


class TextDataset(base_dataset.BaseADDataset):

    def __init__(self, filepath):
        super().__init__(root=filepath)

        # self.normal_classes = tuple([normal_class])
        self.n_classes = 2  # 0: normal, 1: outlier
        # Load your dataset
        df = pd.read_csv(filepath)
        # df['label'] = df['label'].apply(lambda x: int(x in self.normal_classes))

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = AutoModelForCausalLM.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Split the dataset into train and test
        self.train_df, self.test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Create datasets
        self.train_set = self.create_dataset(df, method='sentence-embedding')
        self.test_set = self.create_dataset(self.test_df, method='sentence-embedding')


    def create_dataset(self, df, method='none'):
        # Convert texts and labels into tensors
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        if method == 'word-embedding':
            # Tokenize the texts and convert them to embeddings
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                inputs = self.model(**inputs).last_hidden_state
        elif method == 'sentence-embedding':
            # Convert sentences to embeddings
            with torch.no_grad():
                inputs = self.sentence_embedding_model.encode(texts, convert_to_tensor=True)
        elif method == 'none':
            # Convert sentences to embeddings
            with torch.no_grad():
                inputs = texts
        else:
            raise ValueError('Invalid method: choose either "tokenizer", "word-embedding" or "sentence-embedding"')

        # Return a dictionary with inputs and labels
        # dataset = {'inputs': inputs, 'labels': labels}
        dataset = MyTextDataset({'inputs': inputs, 'labels': labels})
        return dataset

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) \
            -> (DataLoader, DataLoader):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, test_loader

    def __getitem__(self, index):
        inputs = {key: val[index] for key, val in self.train_set['inputs'].items()}
        label = self.train_set['labels'][index]
        return inputs, label, index

    def __len__(self):
        return len(self.train_set['inputs']['input_ids'])


class MyTextDataset(Dataset):
    def __init__(self, data):
        self.inputs = data['inputs']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], idx