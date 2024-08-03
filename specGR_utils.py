import numpy as np
import json
import torch
from torch.utils.data import Dataset
import yaml
import re
from datasets import load_dataset
import sys

class Logger:
    def __init__(self, filename, mode='a'):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
def load_yaml(yaml_file):
    ## The original yaml.safe_load will parse 1e12 as string
    ## This modified version will correctly parse numbers in scientific notation into floats
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    
    with open(yaml_file, 'r') as f:
        config = yaml.load(f, Loader=loader)
    
    return config

def load_dataset_splits(domain, splits):
    data_files = {split: f'{domain}.{split}.csv' for split in splits}
    datasets = load_dataset(f'dataset/{domain}', data_files = data_files)
    return datasets
        
def unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, index

def torch_in(query_tensor, reference_tensor):
    """
    Check if each row in `query_tensor` is in `reference_tensor`.

    Args:
    query_tensor (torch.Tensor): Tensor of shape (N, D) where N is the number of queries,
                                 and D is the dimensionality of each query.
    reference_tensor (torch.Tensor): Tensor of shape (M, D) where M is the number of reference rows,
                                     and D is the dimensionality of each item.

    Returns:
    torch.Tensor: A boolean tensor of shape (N,) where each element is True if the corresponding
                  row in `query_tensor` is in `reference_tensor`.
    """
    # Broadcast query_tensor to shape (N, 1, D) and reference_tensor to (1, M, D)
    query_tensor = query_tensor.unsqueeze(1)       # Shape becomes (N, 1, D)
    reference_tensor = reference_tensor.unsqueeze(0)  # Shape becomes (1, M, D)

    # Compare all elements in the last dimension
    matches = (query_tensor == reference_tensor).all(dim=2)  # Shape becomes (N, M)

    # Check if any of the reference items match the query items
    exists = matches.any(dim=1)  # Shape becomes (N,)

    return exists

def safe_topk(tensor, k, dim=-1):
    """
    A safe version of torch.topk that handles cases where k is greater than the dimension size.
    
    Args:
    tensor (torch.Tensor): The input tensor from which to retrieve the top k elements.
    k (int): The number of top elements to retrieve.
    dim (int, optional): The dimension to retrieve the elements from. Default is -1 (the last dimension).
    
    Returns:
    torch.Tensor, torch.Tensor: The top k values and their corresponding indices.
    """
    if tensor.numel() == 0: return tensor, torch.empty(0, dtype=torch.long).to(tensor.device)

    # If k is within the bounds of the tensor's dimension, use torch.topk. Else use sort.
    if k > tensor.size(dim):
        return torch.sort(tensor, dim=dim, descending=True)
    else:
        return torch.topk(tensor, k, dim=dim)

def gather_indicies(output, gather_index):
    """Gathers the vectors at the specific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)

def load_semantic_id_mappings(config, id_path=None):
    
    domain = config['dataset']
    exp_id = config['exp_id']
    unseen_start_index = config['unseen_start_index']
    
    # if 'map_unseen_semantic_prefix' not in config:
    #     id_path = f"dataset/{domain}/{domain}{exp_id}.semantic_id"
    # else:
    #     id_path = f"dataset/{domain}/{domain}{exp_id}_{config['map_unseen_semantic_prefix']}.semantic_id"
        
    id_path = id_path or f"dataset/{domain}/{domain}{exp_id}.semantic_id"
    
    semantic_ids = np.fromfile(id_path, dtype=np.int64).reshape(-1, 4)
    assert np.all(semantic_ids < config['RQ-VAE']["code_book_size"]), f"Check the Semantic Ids at {id_path}!"
    
    max_num_collison = np.max(semantic_ids[:,3])
    print(f"Number of unique IDs at different code level: 1: {len(np.unique(semantic_ids[:,0]))}, 2: {len(np.unique(semantic_ids[:,1]))}, 3: {len(np.unique(semantic_ids[:,2]))}")
    print(f"Number of max collison at code level 4: {max_num_collison}")
    
    semantic_ids = semantic_ids + (np.arange(4) * config['RQ-VAE']["code_book_size"] + 1).reshape(1,-1)
    
    print('Semantic ids loaded from:', id_path)
    print(semantic_ids)
    
    item_2_semantic_id = {(i+1):list(semantic_ids[i,:]) for i in range(len(semantic_ids))}
    semantic_id_2_item = {tuple(semantic_ids[i,:]):(i+1) for i in range(len(semantic_ids))}

    semantic_prefix_2_items = {}

    for k in range(1,4): # 1,2,3
        k_prefix_2_items = {}
        k_prefix_2_unseen_items = {}
        for i in range(len(semantic_ids)):
            prefix = tuple(semantic_ids[i,:k])
            if prefix in k_prefix_2_items:
                k_prefix_2_items[prefix].append(i+1)
            else:
                k_prefix_2_items[prefix] = [i+1]
            if i >= unseen_start_index:
                if prefix in k_prefix_2_unseen_items:
                    k_prefix_2_unseen_items[prefix].append(i+1)
                else:
                    k_prefix_2_unseen_items[prefix] = [i+1]

        semantic_prefix_2_items[k] = {}
        semantic_prefix_2_items[k]['unseen'] = k_prefix_2_unseen_items
        semantic_prefix_2_items[k]['all'] = k_prefix_2_items
    
    return semantic_ids, item_2_semantic_id, semantic_id_2_item, semantic_prefix_2_items

