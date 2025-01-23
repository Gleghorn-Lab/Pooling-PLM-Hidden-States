import torch
import numpy as np
import networkx as nx
from typing import List, Dict
from transformers import AutoModel


class PoolPartiESM2:
    def __init__(self, seqs: List[str], batch_size: int = 4, model_path: str = "Synthyra/ESMplusplus_small"):
        self.seqs = seqs
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).eval()
        self.tokenizer = self.model.tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_importance_scores(self, att: torch.Tensor) -> Dict[int, float]: # (L, L)
        G = nx.from_numpy_array(att.numpy(), create_using=nx.DiGraph)

        if G.number_of_nodes() != att.shape[0]:
            raise Exception(f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {att.shape[0]} tokens.") 
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any attention edges left in the graph.")
        
        page_rank = nx.pagerank(G, alpha=0.85, max_iter=100, weight='weight', personalization=None, nstart=None)
        total = sum(page_rank.values())
        return {k: v / total for k, v in page_rank.items()}
    
    def _pool(self, rep: torch.Tensor, attn: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: # (b, L, d) (b, L, L) (b, L)
        pooled_reps = []
        for r, a, mask in zip(rep, attn, attention_mask):
            scores = self._get_importance_scores(a)
            scores = torch.tensor(np.array(list(scores.values())))
            pooled_rep = torch.sum(r * scores.unsqueeze(-1), dim=0) / mask.sum()
            pooled_reps.append(pooled_rep)
        return torch.stack(pooled_reps)
    
    def embed_seqs(self) -> Dict[str, torch.Tensor]:
        embedding_dict = {}
        with torch.no_grad():
            for i in range(0, len(self.seqs), self.batch_size):
                seqs = self.seqs[i:i+self.batch_size]
                seqs = self.tokenizer(seqs, return_tensors="pt", padding=True)
                input_ids = seqs.input_ids.to(self.device) # (b, L)
                bs, seq_len = input_ids.shape
                attention_mask = seqs.attention_mask.to(self.device) # (b, L)
                out = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
                rep = out.last_hidden_state.float() # (b, L, d)
                rep = rep * attention_mask.unsqueeze(-1)
                att = torch.cat(out.attentions).float() # (b, n_layers, n_heads, L, L)
                att_mask = attention_mask[:, None, None, None, :].expand(bs, 1, 1, seq_len, seq_len)
                att = att * att_mask
                att = att.max(dim=2).values # (b, n_layers, L, L)
                att = att.max(dim=1).values # (b, L, L)
                pooled_rep = self._pool(rep.cpu(), att.cpu(), attention_mask.cpu())
                embedding_dict[seqs] = pooled_rep
        return embedding_dict


if __name__ == "__main__":
    from datasets import load_dataset
    seqs = load_dataset('GleghornLab/optimal_temperature', split='train')['seqs'][:10]
    seqs = [seq[:8] for seq in seqs]
    pool_parti = PoolPartiESM2(seqs)
    embedding_dict = pool_parti.embed_seqs()
    print(len(embedding_dict))

