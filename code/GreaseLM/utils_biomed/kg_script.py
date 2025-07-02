# Get the article text
import os
from glob import glob
import csv
import json


entities_file = "entities.csv"
relations_file = "relations.csv"

text_file_dir = r"C:\Users\fabien\Documents\Arvind\neph_notebooks\data\textbook_txt_files\textbook_txt_files"
#text_file_dir = r"C:\Users\fabien\Documents\Arvind\neph_notebooks\data\article_txt_files\article_txt_files"
files = sorted(glob(os.path.join(text_file_dir, "*")))

text = []
for file in files[:50]:
    fd = open(file, "r", encoding='utf8')
    text.append(fd.read())
article_txt = "\n".join(text)

import nltk
#nltk.download('punkt')

def clean_text(text):
  """Remove section titles and figure descriptions from text"""
  clean = "\n".join([row for row in text.split("\n") if (len(row.split(" "))) > 3 and not (row.startswith("(a)"))
                    and not row.startswith("Figure")])
  return clean

#text = article_txt.split("INTRODUCTION")[1]
ctext = clean_text(article_txt)
sentences = nltk.tokenize.sent_tokenize(ctext)

import hashlib
import requests
from tqdm import tqdm
import time

#Load scispacy entity linker

import spacy
import scispacy
from scispacy.linking import EntityLinker

def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_sm")
    linker = EntityLinker(
        resolve_abbreviations=True,
        name="umls",
        threshold=threshold)
    nlp.add_pipe(linker)
    return nlp, linker


def entity_linking_to_umls(sentence, nlp, linker):
    doc = nlp(sentence)
    entities = doc.ents
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end, 
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
        all_entities_results.append(curr_entities_result)
    return {"entities": all_entities_results, "sentence": sentence}

def process(input):
    nlp, linker = load_entity_linker()
    stmts = input
    entity_list = []
    for sent in tqdm(sentences):
        entity_list.append(entity_linking_to_umls(sent, nlp, linker))
    return entity_list


async def get_entity_list():
    coros = [request_async(s) for s in sentences[:-1]]
    results = await asyncio.gather(*coros)
    return results


entity_list = []
start = time.time()
entity_list = process(sentences)
end = time.time()
print(end - start)

with open("entities.jsonl", 'w') as fout:
    for dic in entity_list:
        print (json.dumps(dic), file=fout)

# entity_list = []
# The last sentence is invalid
    
# for s in tqdm(sentences[:-1]):
# entity_list.append(query_raw(s))

"""
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
candidates = [s for s in parsed_entities if (s.get('entities')) and (len(s['entities']) > 1)]
predicted_rels = []
for c in candidates:
  combinations = itertools.combinations([{'name':x['entity'], 'id':x['entity_id']} for x in c['entities']], 2)
  for combination in list(combinations):
    # TODO: Why is this failing because not finding entity in text. Should maybe include other alias. 
    try:
      ranked_rels = extractor.rank(text=c['text'].replace(",", " "), head=combination[0]['name'], tail=combination[1]['name'])
      # Define threshold for the most probable relation
      if ranked_rels[0][1] > 0.85:
        predicted_rels.append({'head': combination[0]['id'], 'tail': combination[1]['id'], 'type':ranked_rels[0][0], 'source': c['text_sha256']})
    except:
      pass
print("Time for Relation Extraction")
end = time.time()
print(end - start)

writer1 = csv.writer(open(relations_file, 'w', newline=''))
writer1.writerow(["head", "tail", "type", "text_sha256"])
for rel in predicted_rels:
    writer1.writerow([rel['head'], rel['tail'], rel['type'], rel['source']])
"""