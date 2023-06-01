import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
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


class TextDataset(Dataset):

    def __init__(self, filepath, normal_class=0):
        super().__init__()

        self.normal_classes = tuple([normal_class])
        self.n_classes = 2  # 0: normal, 1: outlier

        # Load your dataset
        df = pd.read_csv(filepath)
        df['label'] = df['label'].apply(lambda x: int(x in self.normal_classes))

        # Tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large')
        # Split the dataset into train and test
        self.train_df, self.test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Create datasets
        self.train_set = self.create_dataset(self.train_df)
        self.test_set = self.create_dataset(self.test_df)

        self.sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_dataset(self, df, method='tokenizer'):
        # Convert texts and labels into tensors
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        if method == 'tokenizer':
            # Tokenize the texts
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        elif method == 'word-embedding':
            # Tokenize the texts and convert them to embeddings
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                inputs = self.model(**inputs).last_hidden_state
        elif method == 'sentence-embedding':
            # Convert sentences to embeddings
            with torch.no_grad():
                inputs = self.sentence_embedding_model.encode(texts, convert_to_tensor=True)
                inputs = {'input_ids': inputs}
        else:
            raise ValueError('Invalid method: choose either "tokenizer", "word-embedding" or "sentence-embedding"')

        # Return a dictionary with inputs and labels
        dataset = {'inputs': inputs, 'labels': labels}
        return dataset

    def __getitem__(self, index):
        inputs = {key: val[index] for key, val in self.train_set['inputs'].items()}
        label = self.train_set['labels'][index]
        return inputs, label, index

    def __len__(self):
        return len(self.train_set['inputs']['input_ids'])
