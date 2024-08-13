from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
import numpy as np
import torch

class Retriever:

    def __init__(self, docs, doc_maxlen, query_maxlen, nbits, colbert_model_name):

        config = ColBERTConfig(doc_maxlen=doc_maxlen, query_maxlen=query_maxlen, nbits=nbits, kmeans_niters=8)
        self.checkpoint = Checkpoint(colbert_model_name, colbert_config=config, verbose=0)
        self.doc_embeds = [self.checkpoint.docFromText(docs_, bsize=8)[0].to('cpu') for docs_ in docs]
        self.docs = docs

    def get_doc_candidates(self, query, top_k):
        q_embed = self.checkpoint.queryFromText([query], bsize=8)[0].to('cpu').to(torch.float16)
        final_scores = [torch.sum(torch.max(torch.matmul(q_embed.squeeze(0), doc_embed.transpose(1, 2)), dim=2)[0], dim=1) for doc_embed in self.doc_embeds]
        para_idxs = [torch.topk(sublist, min(top_k, len(sublist))).indices.tolist() for sublist in final_scores]
        paras = [[self.docs[i][idx] for idx in l] for i, l in enumerate(para_idxs)]
        max_scores = np.array([sublist[para_idxs[i][0]].item() for i, sublist in enumerate(final_scores)])
        return paras, max_scores

    def retrieve(self, query, top_k, doc_num):
        q_embed = self.checkpoint.queryFromText([query], bsize=8)[0].to('cpu').to(torch.float16)
        final_scores = torch.sum(torch.max(torch.matmul(q_embed.squeeze(0), self.doc_embeds[doc_num].transpose(1, 2)), dim=2)[0], dim=1)
        para_idxs = torch.topk(final_scores, min(top_k, len(final_scores))).indices.tolist()
        paras = [self.docs[doc_num][idx] for idx in para_idxs]
        return paras