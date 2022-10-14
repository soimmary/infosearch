from hw4_1.corpus_indexation import tokenize, mean_pooling

from transformers import AutoTokenizer, AutoModel
import torch


def get_query_embedding(
        query: str, model: AutoModel,
        tokenizer: AutoTokenizer, device: torch.device
) -> list:

    enc = tokenize(query, tokenizer, device)
    with torch.no_grad():
        emb = model(**enc)
    semb = mean_pooling(emb, enc['attention_mask'])
    return semb.detach().cpu().numpy()
