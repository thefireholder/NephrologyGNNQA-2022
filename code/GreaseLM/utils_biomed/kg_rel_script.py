# Get the article text
import os
from glob import glob
import csv
import hashlib
import requests
import asyncio
from tqdm import tqdm
import nest_asyncio
import time
import json


relations_file = "relations.csv"

with open('entities.jsonl', 'r') as f:
    lines = f.readlines()
    parsed_entities = [json.loads(line) for line in lines]

import zero_shot_re

from transformers import AutoTokenizer
from zero_shot_re import RelTaggerModel, RelationExtractor
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(device)
model = RelTaggerModel.from_pretrained("fractalego/fewrel-zero-shot").to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
relations = ['associated', 'interacts', 'causes', 'treats']
extractor = RelationExtractor(model, tokenizer, relations)
extractor = extractor

import itertools

# Candidate sentence where there is more than a single entity present
start = time.time()
# should be getting results per sentence (as before)
candidates = [s for s in parsed_entities if len(s['entities']) > 0]
predicted_rels = []
for c in candidates:
  # get highest scoring linked result (for now). Also possible: get relations b/t all possible linkings
  # get all entity combos in sentence
  combos = []
  for x in c["entities"]:
     name = x['text']
     if len(x['linking_results']) > 0:
       id = sorted(x['linking_results'], key=lambda y: y['Score'])[0]["Concept ID"]
     else:
       id = name
     combos.append({"name": name, "id": id})
  combinations = itertools.combinations(combos, 2)
  for combination in list(combinations):
    try:
      ranked_rels = extractor.rank(text=c['sentence'].replace(",", " "), head=combination[0]['name'], tail=combination[1]['name'])
      # Define threshold for the most probable relation
      if ranked_rels[0][1] > 0.85:
        predicted_rels.append({'head': combination[0]['id'], 'tail': combination[1]['id'], 'type':ranked_rels[0][0], 'source': 0})
    except:
      pass
print("Time for Relation Extraction")
end = time.time()
print(end - start)



with open("relations.jsonl", 'w', encoding="utf-8") as fout:
    for dic in predicted_rels:
        print (json.dumps(dic), file=fout)

writer1 = csv.writer(open(relations_file, 'w', newline='', encoding="utf-8"))
writer1.writerow(["head", "tail", "type", "text_sha256"])
for rel in predicted_rels:
    writer1.writerow([rel['head'], rel['tail'], rel['type'], rel['source']])
