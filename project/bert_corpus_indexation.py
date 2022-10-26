from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import json
import numpy as np


# Mean Pooling - take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask) -> float:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Tokenize corpus
def tokenize(corpus: list, tokenizer: AutoTokenizer, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: t.to(device) for k, t in
        tokenizer(
            corpus, padding=True, truncation=True, max_length=256, return_tensors='pt'
        ).items()
    }


# Create embeddings using batches
def get_corpus_embeddings(
        corpus: list, model: AutoModel,
        tokenizer: AutoTokenizer, device: torch.device
) -> list:

    step_size = 256
    max_len = len(corpus)
    n_steps = max_len // step_size + (max_len % step_size > 0)

    sentence_embeddings = []
    for i in tqdm(range(n_steps)):
        start = i * step_size
        stop = min(max_len, (i + 1) * step_size)

        enc = tokenize(corpus[start:stop], tokenizer, device)
        with torch.no_grad():
            emb = model(**enc)
        semb = mean_pooling(emb, enc['attention_mask'])
        sentence_embeddings.append(semb.detach().cpu().numpy().astype(np.float32))
    sentence_embeddings = np.concatenate(sentence_embeddings)
    np.save('corpus_embeddings.npy', sentence_embeddings)
    return sentence_embeddings
