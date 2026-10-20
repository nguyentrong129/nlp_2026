"""Preprocessing / feature extraction cho LAB 04 (IMDb sentiment).

Mọi bước fit (TF-IDF, Word2Vec) chỉ dùng TRAIN, sau đó transform Validation/Test
để tránh data leakage. Code được trích từ classification.ipynb.
"""
from collections import Counter

import numpy as np
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def load_imdb(seed=42, valid_size=0.2):
    """Tải IMDb, chia train/validation (stratified) từ tập train gốc; test giữ nguyên."""
    ds = load_dataset("stanfordnlp/imdb")
    split = ds["train"].train_test_split(
        test_size=valid_size, seed=seed, stratify_by_column="label"
    )
    return split["train"], split["test"], ds["test"]


def inspect_dataset(dataset):
    """Kiểm tra missing text/label, duplicate, độ dài, phân phối lớp."""
    texts = dataset["text"]
    lengths = [len(t.split()) for t in texts]
    return {
        "n_samples": len(texts),
        "class_distribution": Counter(dataset["label"]),
        "missing_text": sum(t is None or t.strip() == "" for t in texts),
        "missing_label": sum(y is None for y in dataset["label"]),
        "duplicates": len(texts) - len(set(texts)),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": sum(lengths) / len(lengths),
    }


def build_tfidf(train_texts, ngram_range=(1, 1), max_features=10000):
    """Fit TF-IDF trên TRAIN. Trả về vectorizer đã fit và ma trận train."""
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = tfidf.fit_transform(train_texts)
    return tfidf, X_train


def train_word2vec(train_texts, vector_size=100, window=5, min_count=2, seed=42):
    """Huấn luyện Word2Vec chỉ trên văn bản TRAIN."""
    sentences = [t.lower().split() for t in train_texts]
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=seed,
    )


def avg_vec(text, w2v):
    """Document vector = trung bình vector các từ có trong vocabulary Word2Vec."""
    vecs = [w2v.wv[w] for w in text.lower().split() if w in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size)
    return np.mean(vecs, axis=0)


def to_avg_w2v(texts, w2v):
    return np.array([avg_vec(t, w2v) for t in texts])
