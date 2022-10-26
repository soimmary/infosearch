import json
from tqdm import tqdm


# Getting texts from jsonl file and building a corpus
def get_texts(path: str) -> list:
    corpus = []
    with open(path, 'r') as f:
        raw_data = list(f)[:50000]
    for item in tqdm(raw_data, desc='Создаю документы'):
        answers = json.loads(item)['answers']
        if answers:
            sorted_answers = sorted((d for d in answers if d['author_rating']['value'] != ''),
                                    key=lambda d: int(d['author_rating']['value']), reverse=True)
            corpus.append(sorted_answers[0]['text'])
    return corpus


corpus = get_texts('data.jsonl')
