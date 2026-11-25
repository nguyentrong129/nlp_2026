"""Reusable SQuAD 1.1 helpers for extractive question answering.

The training notebook uses these functions directly, without a QA pipeline.
Offsets are half-open character spans in the original context.
"""

import re
import string
from collections import Counter, defaultdict

import numpy as np


def normalize_answer(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_scores(prediction, gold_answers):
    """Return best EM and token F1 over a sample's reference answers."""
    if not gold_answers:
        gold_answers = [""]
    exact, f1 = 0.0, 0.0
    for gold in gold_answers:
        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        sample_em = float(pred_tokens == gold_tokens)
        if not pred_tokens or not gold_tokens:
            sample_f1 = float(pred_tokens == gold_tokens)
        elif not overlap:
            sample_f1 = 0.0
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(gold_tokens)
            sample_f1 = 2 * precision * recall / (precision + recall)
        exact = max(exact, sample_em)
        f1 = max(f1, sample_f1)
    return exact, f1


def score_predictions(examples, predictions):
    rows = []
    for example in examples:
        answer = predictions.get(example["id"], "")
        em, f1 = qa_scores(answer, example["answers"]["text"])
        rows.append({"id": example["id"], "exact_match": em, "f1": f1})
    return rows


def _tokenize(examples, tokenizer, max_length, stride):
    return tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


def prepare_train_features(examples, tokenizer, max_length=384, stride=128):
    """Map answer character offsets to each window's token start/end labels."""
    tokenized = _tokenize(examples, tokenizer, max_length, stride)
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offsets = tokenized.pop("offset_mapping")
    starts, ends = [], []
    for feature_index, sample_index in enumerate(sample_mapping):
        input_ids = tokenized["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(feature_index)
        context_tokens = [i for i, part in enumerate(sequence_ids) if part == 1]
        if not context_tokens:
            starts.append(cls_index)
            ends.append(cls_index)
            continue

        answers = examples["answers"][sample_index]
        if not answers["answer_start"]:
            starts.append(cls_index)
            ends.append(cls_index)
            continue
        char_start = answers["answer_start"][0]
        char_end = char_start + len(answers["text"][0])
        first, last = context_tokens[0], context_tokens[-1]
        if offsets[feature_index][first][0] > char_start or offsets[feature_index][last][1] < char_end:
            starts.append(cls_index)
            ends.append(cls_index)
            continue
        start_token = next(
            (i for i in context_tokens if offsets[feature_index][i][0] <= char_start < offsets[feature_index][i][1]),
            None,
        )
        end_token = next(
            (i for i in reversed(context_tokens) if offsets[feature_index][i][0] < char_end <= offsets[feature_index][i][1]),
            None,
        )
        if start_token is None or end_token is None or start_token > end_token:
            starts.append(cls_index)
            ends.append(cls_index)
        else:
            starts.append(start_token)
            ends.append(end_token)
    tokenized["start_positions"] = starts
    tokenized["end_positions"] = ends
    return tokenized


def prepare_eval_features(examples, tokenizer, max_length=384, stride=128):
    tokenized = _tokenize(examples, tokenizer, max_length, stride)
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = [examples["id"][i] for i in sample_mapping]
    tokenized["offset_mapping"] = [
        [offset if tokenized.sequence_ids(feature_index)[i] == 1 else None
         for i, offset in enumerate(offsets)]
        for feature_index, offsets in enumerate(tokenized["offset_mapping"])
    ]
    return tokenized


def postprocess_predictions(examples, features, raw_predictions, n_best=20, max_answer_length=30):
    """Choose a legal context span across all overlapping windows per example."""
    start_logits, end_logits = raw_predictions
    features_by_id = defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_by_id[feature["example_id"]].append(feature_index)
    predictions, spans = {}, {}
    for example in examples:
        best_score = -float("inf")
        best_text, best_span = "", (None, None)
        for feature_index in features_by_id[example["id"]]:
            offsets = features[feature_index]["offset_mapping"]
            context_indices = [i for i, offset in enumerate(offsets) if offset is not None]
            top_starts = sorted(context_indices, key=lambda i: start_logits[feature_index][i], reverse=True)[:n_best]
            top_ends = sorted(context_indices, key=lambda i: end_logits[feature_index][i], reverse=True)[:n_best]
            for start in top_starts:
                for end in top_ends:
                    if (start > end or end - start + 1 > max_answer_length
                            or offsets[start] is None or offsets[end] is None):
                        continue
                    char_start, char_end = offsets[start][0], offsets[end][1]
                    score = float(start_logits[feature_index][start] + end_logits[feature_index][end])
                    if score > best_score:
                        best_score = score
                        best_span = (char_start, char_end)
                        best_text = example["context"][char_start:char_end]
        predictions[example["id"]] = best_text
        spans[example["id"]] = best_span
    return predictions, spans


STOPWORDS = set("what which who when where why how is are was were did do does the a an of in on at to for from with by and or".split())


def keyword_baseline(question, context):
    """Small sentence overlap heuristic; returns a context substring."""
    question_words = set(re.findall(r"\b\w+\b", question.lower())) - STOPWORDS
    sentence_matches = list(re.finditer(r"[^.!?]+[.!?]?", context))
    if not sentence_matches:
        return ""
    best = max(sentence_matches, key=lambda match: len(question_words & set(re.findall(r"\b\w+\b", match.group().lower()))))
    words = list(re.finditer(r"\b[\w'-]+\b", best.group()))
    if not words:
        return ""
    # Prefer a short span outside the words already present in the question.
    start_index = next((i for i, word in enumerate(words)
                        if word.group().lower() not in question_words | STOPWORDS), 0)
    chosen = words[start_index:start_index + 3]
    return context[best.start() + chosen[0].start():best.start() + chosen[-1].end()]
