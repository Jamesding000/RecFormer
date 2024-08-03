from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
from recformer import RecformerTokenizer
import torch
import unicodedata
import random

from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
from recformer import RecformerTokenizer
import torch
import unicodedata
import random

@dataclass
class FinetuneDataCollatorWithPadding:

    tokenizer: RecformerTokenizer
    tokenized_items: Dict

    def __call__(self, batch):

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        '''

        # print(batch)
        
        batch_item_seq, labels = self.prepare_train_data(batch) #self.sample_train_data(batch_item_ids)
        # print(batch_item_seq)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)
        batch["labels"] = labels

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        
        return batch


    def prepare_train_data(self, batch_item_ids):

        batch_item_seq = [row['items'] for row in batch_item_ids]
        labels = [row['label'] for row in batch_item_ids]

        return batch_item_seq, labels

    def extract_features(self, batch_item_seq):

        # print(self.tokenized_items)

        # print(item)

        features = []

        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids = self.tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids])
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):
        
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features


@dataclass
class EvalDataCollatorWithPadding:

    tokenizer: RecformerTokenizer
    tokenized_items: Dict

    def __call__(self, batch_data):

        '''
        features: A batch of list of item ids
        1. sample training pairs
        2. convert item ids to item features
        3. mask tokens for mlm

        input_ids: (batch_size, seq_len)
        item_position_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        global_attention_mask: (batch_size, seq_len)
        '''
        
        batch_item_seq, labels = self.prepare_eval_data(batch_data)
        batch_feature = self.extract_features(batch_item_seq)
        batch_encode_features = self.encode_features(batch_feature)
        batch = self.tokenizer.padding(batch_encode_features, pad_to_max=False)

        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)

        labels = torch.LongTensor(labels)
        
        return batch, labels

    def prepare_eval_data(self, batch_data):

        batch_item_seq = []
        labels = []

        for data_line in batch_data:

            item_ids = data_line['items']
            label = data_line['label']
            
            batch_item_seq.append(item_ids)
            labels.append(label)

        return batch_item_seq, labels


    def extract_features(self, batch_item_seq):

        features = []

        for item_seq in batch_item_seq:
            feature_seq = []
            for item in item_seq:
                input_ids, token_type_ids = self.tokenized_items[item]
                feature_seq.append([input_ids, token_type_ids])
            features.append(feature_seq)

        return features

    def encode_features(self, batch_feature):
        
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=False))

        return features