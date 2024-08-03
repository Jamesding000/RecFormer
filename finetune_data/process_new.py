import json
from collections import defaultdict
import gzip
import random
from tqdm import tqdm
import argparse
import os
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
    
parser.add_argument('-d', default='Video_Games', help='Dataset Name')
args = parser.parse_args()

domain = args.d


file_path = f'dataset/{domain}/{domain}.data_maps'
with open(file_path, 'r') as file:
    data_maps = json.load(file)

item2id = {}

for k,v in data_maps['item2id'].items():
    if k != '[PAD]':
        item2id[k] = v - 1

meta_dataset = load_dataset(
    'McAuley-Lab/Amazon-Reviews-2023',
    f'raw_meta_{domain}',
    split='full',
    trust_remote_code=True
)

meta_dataset = meta_dataset.filter(lambda t: t['parent_asin'] in item2id)
print(f'{len(meta_dataset)} of {len(item2id) - 1} items have meta data.')

meta_data = dict()
for line in tqdm(meta_dataset):
    # line = json.loads(line)
    attr_dict = dict()
    asin = line['parent_asin']
    category = ' '.join(line['categories'])
    title = line['title']
    attr_dict['title'] = title
    attr_dict['brand'] = ""
    attr_dict['category'] = category
    meta_data[asin] = attr_dict

output_path = f"finetune_data/{domain}"
if not os.path.exists(output_path):
    os.mkdir(output_path)

meta_file = os.path.join(output_path, 'meta_data.json')
smap_file = os.path.join(output_path, 'smap.json')

f_s = open(smap_file, 'w', encoding='utf8')
json.dump(item2id, f_s)
f_s.close()

meta_f = open(meta_file, 'w', encoding='utf8')
json.dump(meta_data, meta_f)
meta_f.close()
