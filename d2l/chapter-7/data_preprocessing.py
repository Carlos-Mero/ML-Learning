import torch
import collections
from torch.utils.data import Dataset
import re

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reversed_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(
            counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = list(sorted(set(['<unk>'] + reversed_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
    def unk(self):
        return self.token_to_idx['<unk>']

class TimeMachine(Dataset):
    def _get_data(self):
        with open("../data/timemachine.txt") as f:
            return f.read()
    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()
    def _tokenize(self, text):
        return list(text)
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, index):
        return (self.X[index], self.Y[index])
    def __init__(self, num_steps):
        super().__init__()
        self.raw_text = self._get_data()
        self.raw_text = self._preprocess(self.raw_text)
        self.tokens = self._tokenize(self.raw_text)
        self.vocab = Vocab(self.tokens)
        self.corpus = [self.vocab[token] for token in self.tokens]
        array = torch.tensor([self.corpus[i:i+num_steps]
                              for i in range(len(self.corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]
