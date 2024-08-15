import datasets
import os
class ConflictDataset():

    """ Initialize the dataset """
    def __init__(self, split, doc_span):
        assert doc_span > 0, "Invalid doc span"
    
        with open('ds.pkl', 'rb') as handle:
            self.ds = pickle.load(handle)
            
        self.queries = self.ds['query']
        self.docs = self.ds['doc_texts']

    def get_item(self, idx):
        return self.queries[idx], self.docs[idx]

    def length(self):
        return len(self.queries)
