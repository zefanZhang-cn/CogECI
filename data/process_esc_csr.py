import os
import pickle
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import LongformerTokenizer, LongformerModel
import csv
import json
# from transformers import LlamaTokenizer, LlamaModel
class ESC_features(object):
    def __init__(self, topic_id, doc_id,
                 enc_text, enc_tokens, sentences,
                 enc_input_ids, enc_mask_ids, node_event,
                 t1_pos, t2_pos, target, rel_type, event_pairs
                 ):
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.enc_text = enc_text
        self.enc_tokens = enc_tokens
        self.sentences = sentences

        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.node_event = node_event
        self.t1_pos = t1_pos
        self.t2_pos = t2_pos
        self.target = target
        self.rel_type = rel_type
        self.event_pairs = event_pairs
class ESC_dataset(Dataset):
    def __init__(self, features):

        self.features = features
        base_data_dir = os.environ.get("COGECI_DATA_DIR", "./data")
        self.event_index_file = os.path.join(base_data_dir, "event_index.json")
        self.data_file_dev = os.path.join(base_data_dir, "ESL_dev_gpt.csv")
        self.data_file_train = os.path.join(base_data_dir, "ESL_train_gpt.csv")
        # Optional, user-generated files (can be absent in a public repo).
        # If not found, sentence labels fall back to all zeros.
        self.llm_file_dev = os.path.join(base_data_dir, "examples", "llm", "llama3help_dev_sen.csv")
        self.llm_file_dev_more = os.path.join(base_data_dir, "examples", "llm", "llama3help_dev_sen_more.csv")
        self.llm_file_train = os.path.join(base_data_dir, "examples", "llm", "llama3help_train_sen.csv")
        self.llm_file_train_more = os.path.join(base_data_dir, "examples", "llm", "llama3help_train_sen_more.csv")



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
    def _read_csv_rows_if_exists(self, path):
        if not path or (not os.path.exists(path)):
            return []
        rows = []
        with open(path, mode='r', newline='', errors='ignore') as file:
            database = csv.reader(file)
            for row_index, row in enumerate(database):
                rows.append(row)
        return rows

    def get_sent_ground(self, doc_id, type, sentences, pair_count):
        if type == 'train':
            llm_file = self.llm_file_train
            llm_file_more = self.llm_file_train_more
        else:
            llm_file = self.llm_file_dev
            llm_file_more = self.llm_file_dev_more
        llm_rows = self._read_csv_rows_if_exists(llm_file)
        llm_rows2 = self._read_csv_rows_if_exists(llm_file_more)

        if len(llm_rows) == 0 or len(llm_rows2) == 0:
            zeros = torch.zeros((pair_count, len(sentences)), dtype=torch.long).to('cuda')
            return zeros, zeros.clone()
        llm_sen = []
        for i in range(1, len(llm_rows)):
            if doc_id == llm_rows[i][1]:
                llm_sen.append(llm_rows[i][6])
        llm_sen1 = [item for sublist in llm_sen for item in eval(sublist)]
        all_senlable = []
        for i in range(0, len(llm_sen1)):
            label = []
            if llm_sen1[i] != 'None':
                lst = [int(x) for x in llm_sen1[i].split(",")]
            for x in range(0, len(sentences)):
                if llm_sen1[i] == 'None':
                    label = [0] * len(sentences)
                    continue
                if x in lst:
                    label.append(1)
                else:
                    label.append(0)
            all_senlable.append(label)
        all_senlable = torch.tensor(all_senlable).to('cuda')

        # llm more
        llm_sen2 = []
        for i in range(1, len(llm_rows2)):
            if doc_id == llm_rows2[i][1]:
                llm_sen2.append(llm_rows2[i][6])
        llm_sen2 = [item for sublist in llm_sen2 for item in eval(sublist)]
        all_senlable2 = []
        for i in range(0, len(llm_sen2)):
            label = []
            if llm_sen2[i] != 'None':
                lst = [int(x) for x in llm_sen2[i].split(",")]
            for x in range(0, len(sentences)):
                if llm_sen2[i] == 'None':
                    label = [0] * len(sentences)
                    continue
                if x in lst:
                    label.append(1)
                else:
                    label.append(0)
            all_senlable2.append(label)
        all_senlable2 = torch.tensor(all_senlable2).to('cuda')
        return all_senlable,all_senlable2
    def get_event_index(self,doc_id):
        with open(self.event_index_file, "r") as file:
            data = json.load(file)  # 加载 JSON 文件内容
        # 循环读取每个字典
        for i, item in enumerate(data):
            if item['doc'] == doc_id:
                node_index = item['events_index']
                sentences = item['sentences']
                break
        document = [item for sublist in sentences for item in sublist]
        max_length_long = len(document)
        stride = 4096  # 滑动步长，可以根据需求调整
        if max_length_long > stride:
            print('max_length_long:', max_length_long)
            raise ValueError(f"Document too long: {max_length_long} > stride={stride}")
        chunks = [document[i:i + max_length_long] for i in range(0, len(document), stride)]
        id = 0
        for chunk in chunks:
            # 对每个 chunk 进行编码
            s = " ".join(chunk)
            print(f"self.llama_tokenizer before: {self.llama_tokenizer}")
            arg_1_idx_doc = self.llama_tokenizer.encode_plus(
                s,
                padding=False,
                # truncation=True,
                max_length=max_length_long,
                return_tensors="pt",
                add_special_tokens=False,
                pad_to_max_length=False,
                return_attention_mask=True,
            )
            if id == 0:
                doc_input_ids = arg_1_idx_doc['input_ids']
                doc_mask_ids = arg_1_idx_doc['attention_mask']
            else:
                doc_input_ids = torch.cat((doc_input_ids, arg_1_idx_doc['input_ids']), dim=0)
                doc_mask_ids = torch.cat((doc_mask_ids, arg_1_idx_doc['attention_mask']), dim=0)
            id += 1


        return doc_input_ids,doc_mask_ids,node_index

    def collate_fn(self, batch):
        enc_input_ids = batch[0].enc_input_ids
        enc_mask_ids = batch[0].enc_mask_ids
        node_event = batch[0].node_event
        t1_pos = [f.t1_pos for f in batch]
        t2_pos = [f.t2_pos for f in batch]
        target = [f.target for f in batch]
        rel_type = [f.rel_type for f in batch]
        event_pairs = batch[0].event_pairs
        doc_id = batch[0].doc_id
        document = batch[0].enc_text
        sentences = batch[0].sentences
        topic_id = batch[0].topic_id
        if int(topic_id)>35:
            set_type = 'dev'
        elif int(topic_id)<=35:
            set_type = 'train'
        else:
            print(topic_id)
            print(dfgkbvhk)
        pair_count = len(target[0]) if len(target) > 0 else 0
        all_senlable,all_senlable2 = self.get_sent_ground(doc_id, set_type, sentences, pair_count)
        return (enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs,all_senlable,all_senlable2,doc_id)
class ESC_processor(object):
    def __init__(self, args, tokenizer, printlog):
        self.args = args
        self.tokenizer = tokenizer
        self.printlog = printlog
    def convert_features_to_dataset(self, features):
        dataset = ESC_dataset(features)
        return dataset

    def load_and_cache_features(self, cache_path):
        features = pickle.load(open(cache_path, 'rb'))
        self.printlog(f"load features from {cache_path}")
        return features

    def generate_dataloader(self, set_type):
        assert (set_type in ['train', 'dev'])
        cache_feature_path = os.path.join(self.args.cache_path,
                                          "{}_{}_{}_{}_features.pkl".format(self.args.dataset_type, self.args.model_type, self.args.inter_or_intra,
                                                                         set_type))

        features = self.load_and_cache_features(cache_feature_path)

        dataset = self.convert_features_to_dataset(features)

        return features, dataset