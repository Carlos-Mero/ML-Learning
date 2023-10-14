import collections
import random
import re
import torch
from d2l import torch as d2l

class TimeMachine(d2l.DataModule):
    """The Time Machine dataset."""
    def _download(self):
        fname=d2l.download(d2l.DATA_URL + 'timemachine.txt',
                           self.root,
                           '090b5e7e70c295757f55df93cb0a180b9691891a')
        print(fname)
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()

@d2l.add_to_class(TimeMachine)
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)

@d2l.add_to_class(TimeMachine)
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
