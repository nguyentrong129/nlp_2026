"""Answer a question from a supplied document with a trained extractive QA model.

Example:
    python document_qa.py --context-file article.txt --question "Who wrote it?"
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from qa_model import postprocess_predictions, prepare_eval_features


DEFAULT_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


def answer_question(question, context, model_name=DEFAULT_MODEL, max_length=384, stride=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    example = {"id": "document-1", "question": question, "context": context}
    batch = {key: [value] for key, value in example.items()}
    feature_dict = prepare_eval_features(batch, tokenizer, max_length, stride)
    features = [{key: value[i] for key, value in feature_dict.items()}
                for i in range(len(feature_dict["input_ids"]))]
    start_logits, end_logits = [], []
    with torch.no_grad():
        for feature in features:
            inputs = {
                key: torch.tensor([feature[key]])
                for key in ("input_ids", "attention_mask", "token_type_ids") if key in feature
            }
            output = model(**inputs)
            start_logits.append(output.start_logits[0].cpu().numpy())
            end_logits.append(output.end_logits[0].cpu().numpy())
    answers, spans = postprocess_predictions(
        [example], features, (np.stack(start_logits), np.stack(end_logits))
    )
    return answers[example["id"]], spans[example["id"]]


def main():
    parser = argparse.ArgumentParser(description="Extract an answer from a supplied text document")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="QA model name on Hugging Face or a local model directory")
    parser.add_argument("--context-file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()
    context = Path(args.context_file).read_text(encoding="utf-8")
    answer, (start, end) = answer_question(
        args.question, context, args.model, args.max_length, args.stride
    )
    print("Answer:", answer)
    print("Character span:", (start, end))


if __name__ == "__main__":
    main()
