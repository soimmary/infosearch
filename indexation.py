from hw1.directories import get_directories
from sklearn.feature_extraction.text import CountVectorizer


def create_corpus(main_dir: str) -> list:
    corpus = []
    dirs = get_directories(main_dir)
    for dir in dirs:
        with open(dir, 'r', encoding='utf-8') as f:
            text = f.read()
            corpus.append({dir: text.split()})
            f.close()
    return corpus


def count_lemma(tokens: list, lemma: str) -> int:
    return sum([token == lemma for token in tokens])


def create_index_dict(main_dir: str) -> dict:
    corpus = create_corpus(main_dir)
    new_dic = {}
    for item in corpus:
        for dir, text in item.items():
            for lemma in text:
                num = count_lemma(text, lemma)
                new_dic.setdefault(lemma, []).append([dir.split('/')[2], num])
    return new_dic


def create_index_matrix(main_dir: str) -> tuple:
    corpus = create_corpus(main_dir)
    corpus = [' '.join(list(text.values())[0]) for text in corpus]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    lemmas = vectorizer.get_feature_names_out()
    matrix = X.toarray()
    return (lemmas, matrix)
