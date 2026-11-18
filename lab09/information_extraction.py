# Information Extraction using the fine-tuned LAB09 token classifier
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR="lab09_outputs/bert_ner"

def load_ner():
    return (AutoTokenizer.from_pretrained(MODEL_DIR),
            AutoModelForTokenClassification.from_pretrained(MODEL_DIR))

def bio_to_entities(words,tags):
    result=[]; start=None; kind=None
    for i,tag in enumerate(tags+["O"]):
        if tag=="O" or tag.startswith("B-"):
            if start is not None:
                result.append({"text":" ".join(words[start:i]),"type":kind})
            start=None; kind=None
            if tag.startswith("B-"): start=i; kind=tag[2:]
        elif tag.startswith("I-") and (start is None or kind!=tag[2:]):
            if start is not None:
                result.append({"text":" ".join(words[start:i]),"type":kind})
            start=i; kind=tag[2:]
    return result

def extract(text,tokenizer,model):
    words=text.split()
    enc=tokenizer(words,is_split_into_words=True,return_tensors="pt",truncation=True)
    model.eval()
    with torch.no_grad(): ids=model(**enc).logits.argmax(-1)[0].tolist()
    word_ids=tokenizer(words,is_split_into_words=True,truncation=True).word_ids()
    tags={}
    for wid,pid in zip(word_ids,ids):
        if wid is not None and wid not in tags:
            tags[wid]=model.config.id2label[pid]
    entities=bio_to_entities(words,[tags.get(i,"O") for i in range(len(words))])
    output={"PERSON":[],"ORG":[],"LOCATION":[],"MISC":[]}
    for e in entities:
        output[{"PER":"PERSON","ORG":"ORG","LOC":"LOCATION","MISC":"MISC"}[e["type"]]].append(e["text"])
    return output

if __name__=="__main__":
    tok,model=load_ner()
    print(extract("Barack Obama visited Paris.",tok,model))
